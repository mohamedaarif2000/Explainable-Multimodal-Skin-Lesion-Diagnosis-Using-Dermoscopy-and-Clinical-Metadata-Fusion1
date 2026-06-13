"""
RQ4 - How effectively can explainable AI techniques (Grad-CAM and SHAP) improve
transparency and trustworthiness of multimodal skin lesion diagnosis systems?

Evaluation: Visual explanation quality (Grad-CAM explanation consistency, SHAP
feature-importance stability) and dermatologist assessment of case-study
explanations.

Requires: pip install shap opencv-python scipy
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from scipy.stats import spearmanr

try:
    import shap
except ImportError as exc:
    raise ImportError("The 'shap' package is required for RQ4. Install it with: pip install shap") from exc

from config import DATA_ROOT, DEVICE, EPOCHS, SEED, SHAP_BACKGROUND_SIZE, SHAP_N_SAMPLES, make_output_dirs
from data_pipeline import build_transforms, load_data
from engine import evaluate_model, train_model
from models import AttentionFusionModel
from plotting import plot_confusion_matrix, plot_training_curves

RQ_TAG = "rq4_explainability_analysis"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denorm_image(img_tensor):
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, img, meta, class_idx):
        self.model.zero_grad()
        out = self.model(img, meta)
        score = out[:, class_idx].sum()
        score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1))
        cam = cam[0].cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def make_meta_predict_fn(model, i_proj, class_idx):
    def predict_fn(meta_array):
        meta_t = torch.tensor(meta_array, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            out = model.forward_from_image_projection(i_proj, meta_t)
            probs = F.softmax(out, dim=1)
        return probs[:, class_idx].cpu().numpy()
    return predict_fn


def explain_sample_metadata(model, background, img_tensor, meta_tensor, class_idx):
    with torch.no_grad():
        i_proj = model.project_image(img_tensor.unsqueeze(0).to(DEVICE))
    predict_fn = make_meta_predict_fn(model, i_proj, class_idx)
    explainer = shap.KernelExplainer(predict_fn, background, silent=True)
    shap_values = explainer.shap_values(meta_tensor.numpy().reshape(1, -1), nsamples=SHAP_N_SAMPLES, silent=True)
    return np.array(shap_values).reshape(-1)


def select_case_indices(labels_all, preds_all, class_names, n_correct=2, n_incorrect=2):
    correct_idx = np.where(labels_all == preds_all)[0]
    incorrect_idx = np.where(labels_all != preds_all)[0]

    selected = []
    if "MEL" in class_names:
        mel_label = class_names.index("MEL")
        mel_idx = np.where(labels_all == mel_label)[0]
        for pool in [np.intersect1d(mel_idx, correct_idx), np.intersect1d(mel_idx, incorrect_idx)]:
            if len(pool) > 0:
                selected.append(int(pool[0]))

    rng = np.random.RandomState(SEED)
    remaining_correct = [i for i in correct_idx if i not in selected]
    remaining_incorrect = [i for i in incorrect_idx if i not in selected]

    n_corr_needed = max(0, n_correct - sum(1 for i in selected if labels_all[i] == preds_all[i]))
    n_inc_needed = max(0, n_incorrect - sum(1 for i in selected if labels_all[i] != preds_all[i]))

    if len(remaining_correct) > 0 and n_corr_needed > 0:
        selected += list(rng.choice(remaining_correct, size=min(n_corr_needed, len(remaining_correct)), replace=False))
    if len(remaining_incorrect) > 0 and n_inc_needed > 0:
        selected += list(rng.choice(remaining_incorrect, size=min(n_inc_needed, len(remaining_incorrect)), replace=False))

    return selected


def main():
    output_dir, fig_dir, tab_dir, model_dir = make_output_dirs(RQ_TAG)
    print("Outputs will be saved under:", output_dir)

    data = load_data(DATA_ROOT)
    _train_tf, eval_tf = build_transforms()

    model = AttentionFusionModel(data.meta_dim, data.num_classes)
    model, history = train_model(
        model, data.train_loader, data.val_loader, data.class_weights, model_dir,
        epochs=EPOCHS, modality="multimodal", tag="attention_fusion_rq4",
    )
    plot_training_curves(history, "Attention fusion model: training curves",
                          "rq4_training_curves.pdf", fig_dir)

    metrics, labels_all, preds_all, probs_all = evaluate_model(model, data.test_loader, modality="multimodal")
    print("Test metrics:", {k: round(v, 4) for k, v in metrics.items()})
    plot_confusion_matrix(labels_all, preds_all, data.class_names,
                           "Attention fusion model: confusion matrix (test set)",
                           "rq4_confusion_matrix.pdf", fig_dir)

    target_layer = model.img_enc.features[-1]
    gradcam = GradCAM(model, target_layer)

    def gradcam_overlay(img_tensor, meta_tensor, class_idx, img_size=224):
        img_in = img_tensor.unsqueeze(0).to(DEVICE)
        meta_in = meta_tensor.unsqueeze(0).to(DEVICE)
        cam = gradcam.generate(img_in, meta_in, class_idx)
        return cv2.resize(cam, (img_size, img_size))

    default_background = data.train_meta.sample(
        n=min(SHAP_BACKGROUND_SIZE, len(data.train_meta)), random_state=SEED
    ).values.astype(np.float32)
    model.eval()

    def get_attention_weights(img_tensor, meta_tensor):
        img_in = img_tensor.unsqueeze(0).to(DEVICE)
        meta_in = meta_tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, attn_weights = model(img_in, meta_in, return_attention=True)
        w = attn_weights[0].cpu().numpy()
        image_weight = w[:, 0].mean()
        meta_weight = w[:, 1].mean()
        total = image_weight + meta_weight
        return image_weight / total, meta_weight / total

    # --- Explanation Consistency: Grad-CAM stability under perturbation ---
    N_CONSISTENCY_SAMPLES = 30
    rng = np.random.RandomState(SEED)
    consistency_indices = rng.choice(len(data.test_ds), size=min(N_CONSISTENCY_SAMPLES, len(data.test_ds)), replace=False)

    def perturb_image(img_path, transform):
        img = Image.open(img_path).convert("RGB")
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = ImageEnhance.Brightness(img).enhance(1.15)
        return transform(img)

    consistency_rows = []
    for idx in consistency_indices:
        img_t, meta_t, _label_t = data.test_ds[idx]
        pred_class = int(preds_all[idx])

        cam_orig = gradcam_overlay(img_t, meta_t, pred_class)

        img_path = data.test_df.iloc[idx]["img_path"]
        img_pert = perturb_image(img_path, eval_tf)
        cam_pert = gradcam_overlay(img_pert, meta_t, pred_class)

        corr = np.corrcoef(cam_orig.flatten(), cam_pert.flatten())[0, 1]
        consistency_rows.append({"test_index": int(idx), "gradcam_correlation": corr})

    consistency_df = pd.DataFrame(consistency_rows)
    consistency_path = os.path.join(tab_dir, "rq4_gradcam_consistency.csv")
    consistency_df.to_csv(consistency_path, index=False)
    print("Saved", consistency_path)
    print("Mean Grad-CAM consistency (Pearson r):", consistency_df["gradcam_correlation"].mean().round(4))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(consistency_df["gradcam_correlation"], bins=15, edgecolor="black")
    ax.axvline(consistency_df["gradcam_correlation"].mean(), color="red", linestyle="--", label="mean")
    ax.set_xlabel("Pearson correlation between original and perturbed Grad-CAM")
    ax.set_ylabel("Count")
    ax.set_title("RQ4: Grad-CAM explanation consistency under label-preserving perturbation")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "rq4_gradcam_consistency_hist.pdf"), format="pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # --- Feature Importance Stability: SHAP ranking stability ---
    N_STABILITY_SAMPLES = 6
    N_REPEATS = 3

    rng2 = np.random.RandomState(SEED + 1)
    stability_indices = rng2.choice(len(data.test_ds), size=min(N_STABILITY_SAMPLES, len(data.test_ds)), replace=False)

    stability_rows = []
    for idx in stability_indices:
        img_t, meta_t, _label_t = data.test_ds[idx]
        pred_class = int(preds_all[idx])

        rankings = []
        for r in range(N_REPEATS):
            bg = data.train_meta.sample(
                n=min(SHAP_BACKGROUND_SIZE, len(data.train_meta)), random_state=SEED + 10 + r
            ).values.astype(np.float32)
            sv = explain_sample_metadata(model, bg, img_t, meta_t, pred_class)
            rankings.append(np.abs(sv))

        pairwise_corrs = []
        for a in range(N_REPEATS):
            for b in range(a + 1, N_REPEATS):
                corr, _ = spearmanr(rankings[a], rankings[b])
                pairwise_corrs.append(corr)

        stability_rows.append({"test_index": int(idx), "mean_spearman_correlation": np.mean(pairwise_corrs)})

    stability_df = pd.DataFrame(stability_rows)
    stability_path = os.path.join(tab_dir, "rq4_shap_stability.csv")
    stability_df.to_csv(stability_path, index=False)
    print("Saved", stability_path)
    print("Mean SHAP ranking stability (Spearman r):", stability_df["mean_spearman_correlation"].mean().round(4))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(stability_df["test_index"].astype(str), stability_df["mean_spearman_correlation"])
    ax.axhline(stability_df["mean_spearman_correlation"].mean(), color="red", linestyle="--", label="mean")
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("Mean pairwise Spearman correlation")
    ax.set_title("RQ4: SHAP feature-importance ranking stability")
    ax.set_ylim(-1, 1)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "rq4_shap_stability.pdf"), format="pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # --- Case studies: combined Grad-CAM + SHAP + attention-weight explanations ---
    case_indices = select_case_indices(labels_all, preds_all, data.class_names)
    print("Selected test-set indices for case studies:", case_indices)

    case_summaries = []
    TOP_K_FEATURES = 8

    for case_num, idx in enumerate(case_indices, start=1):
        img_t, meta_t, label_t = data.test_ds[idx]
        true_class = data.class_names[int(label_t)]
        pred_class_idx = int(preds_all[idx])
        pred_class = data.class_names[pred_class_idx]
        pred_prob = float(probs_all[idx, pred_class_idx])

        cam = gradcam_overlay(img_t, meta_t, pred_class_idx)
        img_disp = denorm_image(img_t)

        sv = explain_sample_metadata(model, default_background, img_t, meta_t, pred_class_idx)
        sv_df = pd.DataFrame({"feature": data.meta_feature_names, "shap_value": sv})
        sv_df = sv_df.reindex(sv_df["shap_value"].abs().sort_values(ascending=False).index).head(TOP_K_FEATURES)
        sv_df = sv_df.iloc[::-1]

        img_w, meta_w = get_attention_weights(img_t, meta_t)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), gridspec_kw={"width_ratios": [1, 1, 1.3]})

        axes[0].imshow(img_disp)
        axes[0].set_title("Dermoscopic image")
        axes[0].axis("off")

        axes[1].imshow(img_disp)
        axes[1].imshow(cam, cmap="jet", alpha=0.45)
        axes[1].set_title("Grad-CAM (predicted class)")
        axes[1].axis("off")

        colors = ["#d62728" if v < 0 else "#2ca02c" for v in sv_df["shap_value"]]
        axes[2].barh(sv_df["feature"], sv_df["shap_value"], color=colors)
        axes[2].set_xlabel("SHAP value (effect on predicted-class probability)")
        axes[2].set_title(f"Top metadata contributions\nAttention weights: image={img_w:.2f}, metadata={meta_w:.2f}")

        correct_str = "CORRECT" if true_class == pred_class else "INCORRECT"
        fig.suptitle(f"Case {case_num} - True: {true_class} | Predicted: {pred_class} "
                      f"(p={pred_prob:.2f}) - {correct_str}")
        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"rq4_case_{case_num}_combined.pdf"), format="pdf", bbox_inches="tight")
        plt.show()
        plt.close(fig)

        case_summaries.append({
            "case": case_num,
            "test_index": int(idx),
            "true_class": true_class,
            "predicted_class": pred_class,
            "predicted_probability": pred_prob,
            "correct": true_class == pred_class,
            "attention_weight_image": img_w,
            "attention_weight_metadata": meta_w,
        })

    case_summary_df = pd.DataFrame(case_summaries)
    case_summary_path = os.path.join(tab_dir, "rq4_case_studies_summary.csv")
    case_summary_df.to_csv(case_summary_path, index=False)
    print("Saved", case_summary_path)
    print(case_summary_df)

    rating_template = case_summary_df[
        ["case", "test_index", "true_class", "predicted_class", "predicted_probability", "correct"]
    ].copy()
    rating_template["gradcam_plausibility_1to5"] = ""
    rating_template["shap_plausibility_1to5"] = ""
    rating_template["overall_trust_1to5"] = ""
    rating_template["comments"] = ""

    rating_path = os.path.join(tab_dir, "rq4_dermatologist_rating_template.csv")
    rating_template.to_csv(rating_path, index=False)
    print("Saved", rating_path)


if __name__ == "__main__":
    main()