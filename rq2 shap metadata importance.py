"""
RQ2 - Which clinical metadata features contribute most significantly to skin
lesion diagnosis in a multimodal framework?

Evaluation: SHAP feature-importance ranking (global and per diagnostic class).

Requires: pip install shap
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F

try:
    import shap
except ImportError as exc:
    raise ImportError("The 'shap' package is required for RQ2. Install it with: pip install shap") from exc

from config import DATA_ROOT, DEVICE, EPOCHS, SEED, SHAP_BACKGROUND_SIZE, SHAP_N_SAMPLES, make_output_dirs
from data_pipeline import load_data
from engine import evaluate_model, train_model
from models import AttentionFusionModel
from plotting import plot_training_curves

RQ_TAG = "rq2_shap_metadata_importance"


def make_meta_predict_fn(model, i_proj, class_idx):
    """Returns a function: metadata array (N, META_DIM) -> predicted-class probability (N,)"""
    def predict_fn(meta_array):
        meta_t = torch.tensor(meta_array, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            out = model.forward_from_image_projection(i_proj, meta_t)
            probs = F.softmax(out, dim=1)
        return probs[:, class_idx].cpu().numpy()
    return predict_fn


def explain_sample_metadata(model, background_meta, img_tensor, meta_tensor, class_idx):
    with torch.no_grad():
        i_proj = model.project_image(img_tensor.unsqueeze(0).to(DEVICE))
    predict_fn = make_meta_predict_fn(model, i_proj, class_idx)
    explainer = shap.KernelExplainer(predict_fn, background_meta, silent=True)
    shap_values = explainer.shap_values(meta_tensor.numpy().reshape(1, -1), nsamples=SHAP_N_SAMPLES, silent=True)
    return np.array(shap_values).reshape(-1)


def main():
    output_dir, fig_dir, tab_dir, model_dir = make_output_dirs(RQ_TAG)
    print("Outputs will be saved under:", output_dir)

    data = load_data(DATA_ROOT)

    model = AttentionFusionModel(data.meta_dim, data.num_classes)
    model, history = train_model(
        model, data.train_loader, data.val_loader, data.class_weights, model_dir,
        epochs=EPOCHS, modality="multimodal", tag="attention_fusion_rq2",
    )
    plot_training_curves(history, "Attention fusion model: training curves",
                          "rq2_training_curves.pdf", fig_dir)

    metrics, labels_all, preds_all, probs_all = evaluate_model(model, data.test_loader, modality="multimodal")
    print("Test metrics:", {k: round(v, 4) for k, v in metrics.items()})

    background_meta = data.train_meta.sample(
        n=min(SHAP_BACKGROUND_SIZE, len(data.train_meta)), random_state=SEED
    ).values.astype(np.float32)

    model.eval()

    N_GLOBAL_SAMPLES = 30
    rng = np.random.RandomState(SEED)
    global_indices = rng.choice(len(data.test_ds), size=min(N_GLOBAL_SAMPLES, len(data.test_ds)), replace=False)

    all_shap_values = []
    all_pred_classes = []
    for idx in global_indices:
        img_t, meta_t, _label_t = data.test_ds[idx]
        pred_class = int(preds_all[idx])
        sv = explain_sample_metadata(model, background_meta, img_t, meta_t, pred_class)
        all_shap_values.append(sv)
        all_pred_classes.append(pred_class)

    all_shap_values = np.array(all_shap_values)
    all_pred_classes = np.array(all_pred_classes)
    sampled_meta_values = np.stack([data.test_ds[idx][1].numpy() for idx in global_indices])

    mean_abs_shap = np.abs(all_shap_values).mean(axis=0)
    mean_signed_shap = all_shap_values.mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": data.meta_feature_names,
        "mean_abs_shap": mean_abs_shap,
        "mean_signed_shap": mean_signed_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    importance_path = os.path.join(tab_dir, "rq2_shap_global_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print("Saved", importance_path)

    top_n = 15
    fig, ax = plt.subplots(figsize=(8, 6))
    top_features = importance_df.head(top_n).iloc[::-1]
    ax.barh(top_features["feature"], top_features["mean_abs_shap"])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"RQ2: Top {top_n} clinical metadata features by global SHAP importance")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "rq2_shap_global_importance.pdf"), format="pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(importance_df.head(top_n))

    plt.figure(figsize=(9, 8))
    shap.summary_plot(all_shap_values, sampled_meta_values, feature_names=data.meta_feature_names,
                       max_display=15, show=False)
    fig = plt.gcf()
    fig.suptitle("RQ2: SHAP summary plot (clinical metadata, sampled test instances)")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "rq2_shap_summary_beeswarm.pdf"), format="pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)

    per_class_rows = []
    for class_idx, class_name in enumerate(data.class_names):
        mask = all_pred_classes == class_idx
        if mask.sum() == 0:
            continue
        mean_abs = np.abs(all_shap_values[mask]).mean(axis=0)
        for feat, val in zip(data.meta_feature_names, mean_abs):
            per_class_rows.append({"class": class_name, "feature": feat,
                                    "mean_abs_shap": val, "n_samples": int(mask.sum())})

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_path = os.path.join(tab_dir, "rq2_shap_per_class_importance.csv")
    per_class_df.to_csv(per_class_path, index=False)
    print("Saved", per_class_path)

    if len(per_class_df) > 0:
        top_feats = importance_df.head(12)["feature"].tolist()
        pivot = per_class_df[per_class_df["feature"].isin(top_feats)].pivot(
            index="feature", columns="class", values="mean_abs_shap")
        pivot = pivot.reindex(top_feats)

        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
        ax.set_title("RQ2: Per-class SHAP feature importance (top global features)")
        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, "rq2_shap_per_class_heatmap.pdf"), format="pdf", bbox_inches="tight")
        plt.show()
        plt.close(fig)

    print(per_class_df.sort_values(["class", "mean_abs_shap"], ascending=[True, False]).groupby("class").head(5))


if __name__ == "__main__":
    main()