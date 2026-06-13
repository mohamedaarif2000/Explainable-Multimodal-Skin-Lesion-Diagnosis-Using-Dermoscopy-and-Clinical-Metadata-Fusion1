"""
RQ5 - Can uncertainty estimation identify unreliable predictions and improve
the safety of multimodal skin lesion diagnosis?

Evaluation: Monte Carlo Dropout; confidence calibration metrics (Expected
Calibration Error, Brier Score); misclassification-detection AUROC;
risk-coverage analysis.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score

from config import DATA_ROOT, DEVICE, EPOCHS, MC_DROPOUT_T, make_output_dirs
from data_pipeline import load_data
from engine import evaluate_model, train_model
from models import AttentionFusionModel
from plotting import plot_confusion_matrix, plot_training_curves

RQ_TAG = "rq5_uncertainty_estimation"


def enable_mc_dropout(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(model, loader, T=MC_DROPOUT_T):
    model.eval()
    enable_mc_dropout(model)

    all_probs = []
    labels_collected = None

    with torch.no_grad():
        for t in range(T):
            batch_probs = []
            batch_labels = []
            for img, meta, label in loader:
                img, meta = img.to(DEVICE), meta.to(DEVICE)
                out = model(img, meta)
                probs = F.softmax(out, dim=1).cpu().numpy()
                batch_probs.append(probs)
                if t == 0:
                    batch_labels.append(label.numpy())
            all_probs.append(np.concatenate(batch_probs))
            if t == 0:
                labels_collected = np.concatenate(batch_labels)

    all_probs = np.stack(all_probs)  # (T, N, C)
    return all_probs, labels_collected


def expected_calibration_error(labels, preds, confidences, n_bins=10):
    correct = (preds == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_acc, bin_conf, bin_counts = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += abs(acc - conf) * mask.sum()
        bin_acc.append(acc); bin_conf.append(conf); bin_counts.append(mask.sum())
    ece /= len(labels)
    return ece, np.array(bin_acc), np.array(bin_conf), np.array(bin_counts)


def multiclass_brier_score(labels, probs):
    onehot = np.eye(probs.shape[1])[labels]
    return np.mean(np.sum((probs - onehot) ** 2, axis=1))


def reliability_diagram(labels, preds, confidences, title, filename, fig_dir, n_bins=10):
    ece, bin_acc, bin_conf, _bin_counts = expected_calibration_error(labels, preds, confidences, n_bins)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(bin_conf, bin_acc, "o-", label="Model")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, filename), format="pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return ece


def main():
    output_dir, fig_dir, tab_dir, model_dir = make_output_dirs(RQ_TAG)
    print("Outputs will be saved under:", output_dir)

    data = load_data(DATA_ROOT)

    model = AttentionFusionModel(data.meta_dim, data.num_classes)
    model, history = train_model(
        model, data.train_loader, data.val_loader, data.class_weights, model_dir,
        epochs=EPOCHS, modality="multimodal", tag="attention_fusion_rq5",
    )
    plot_training_curves(history, "Attention fusion model: training curves",
                          "rq5_training_curves.pdf", fig_dir)

    single_metrics, labels_all, single_preds, single_probs = evaluate_model(model, data.test_loader, modality="multimodal")
    print("Single-pass metrics:", {k: round(v, 4) for k, v in single_metrics.items()})
    plot_confusion_matrix(labels_all, single_preds, data.class_names,
                           "Attention fusion model: confusion matrix (test set, single pass)",
                           "rq5_confusion_matrix.pdf", fig_dir)

    print(f"Running {MC_DROPOUT_T} stochastic forward passes over the test set...")
    mc_probs, labels_mc = mc_dropout_predict(model, data.test_loader)
    assert np.array_equal(labels_mc, labels_all), "Label order mismatch between single-pass and MC-Dropout evaluation"

    mean_probs = mc_probs.mean(axis=0)
    mc_preds = mean_probs.argmax(axis=1)
    mc_confidence = mean_probs.max(axis=1)

    eps = 1e-12
    predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + eps), axis=1)
    epistemic_uncertainty = mc_probs.var(axis=0).mean(axis=1)

    mc_accuracy = accuracy_score(labels_all, mc_preds)
    print("MC-Dropout (mean) accuracy:", round(mc_accuracy, 4))

    predictions_df = pd.DataFrame({
        "test_index": np.arange(len(labels_all)),
        "true_label": [data.class_names[i] for i in labels_all],
        "predicted_label": [data.class_names[i] for i in mc_preds],
        "correct": labels_all == mc_preds,
        "mc_confidence": mc_confidence,
        "predictive_entropy": predictive_entropy,
        "epistemic_uncertainty": epistemic_uncertainty,
        "single_pass_confidence": single_probs.max(axis=1),
    })

    pred_path = os.path.join(tab_dir, "rq5_predictions_with_uncertainty.csv")
    predictions_df.to_csv(pred_path, index=False)
    print("Saved", pred_path)

    ece_single = reliability_diagram(labels_all, single_preds, single_probs.max(axis=1),
                                      "Single-pass model: calibration", "rq5_calibration_single_pass.pdf", fig_dir)
    ece_mc = reliability_diagram(labels_all, mc_preds, mc_confidence,
                                  "MC-Dropout model: calibration", "rq5_calibration_mc_dropout.pdf", fig_dir)

    brier_single = multiclass_brier_score(labels_all, single_probs)
    brier_mc = multiclass_brier_score(labels_all, mean_probs)

    calibration_summary = pd.DataFrame([
        {"method": "Single-pass", "accuracy": single_metrics["accuracy"], "ece": ece_single, "brier_score": brier_single},
        {"method": "MC-Dropout (mean)", "accuracy": mc_accuracy, "ece": ece_mc, "brier_score": brier_mc},
    ])

    calibration_path = os.path.join(tab_dir, "rq5_calibration_summary.csv")
    calibration_summary.to_csv(calibration_path, index=False)
    print("Saved", calibration_path)
    print(calibration_summary)

    incorrect = (~predictions_df["correct"]).astype(int).values

    auroc_entropy = roc_auc_score(incorrect, predictive_entropy)
    auroc_epistemic = roc_auc_score(incorrect, epistemic_uncertainty)
    auroc_neg_confidence = roc_auc_score(incorrect, -mc_confidence)

    misclass_detection = pd.DataFrame([
        {"uncertainty_measure": "predictive_entropy", "auroc_for_error_detection": auroc_entropy},
        {"uncertainty_measure": "epistemic_uncertainty (MC variance)", "auroc_for_error_detection": auroc_epistemic},
        {"uncertainty_measure": "negative confidence", "auroc_for_error_detection": auroc_neg_confidence},
    ])

    misclass_path = os.path.join(tab_dir, "rq5_misclassification_detection.csv")
    misclass_detection.to_csv(misclass_path, index=False)
    print("Saved", misclass_path)
    print(misclass_detection)

    entropy_bins, _bin_edges = pd.qcut(predictions_df["predictive_entropy"], q=4, duplicates="drop", retbins=True)
    n_actual_bins = entropy_bins.cat.categories.shape[0]
    bin_labels = [f"Q{i+1} (lowest unc.)" if i == 0 else
                  (f"Q{i+1} (highest unc.)" if i == n_actual_bins - 1 else f"Q{i+1}")
                  for i in range(n_actual_bins)]
    predictions_df["entropy_bin"] = entropy_bins.cat.rename_categories(bin_labels)

    bin_acc_df = predictions_df.groupby("entropy_bin", observed=True)["correct"].mean().reset_index()
    bin_acc_df.columns = ["entropy_bin", "accuracy"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(bin_acc_df["entropy_bin"].astype(str), bin_acc_df["accuracy"])
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Predictive-entropy quartile")
    ax.set_title("RQ5: Accuracy by predictive-uncertainty quartile")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "rq5_accuracy_by_uncertainty_bin.pdf"), format="pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(bin_acc_df)

    order = np.argsort(-mc_confidence)
    sorted_correct = predictions_df["correct"].values[order]

    coverages = np.arange(1, len(sorted_correct) + 1) / len(sorted_correct)
    cum_accuracy = np.cumsum(sorted_correct) / np.arange(1, len(sorted_correct) + 1)
    risk = 1 - cum_accuracy

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(coverages, risk)
    ax.set_xlabel("Coverage (fraction of test set retained, most confident first)")
    ax.set_ylabel("Risk (error rate) on retained predictions")
    ax.set_title("RQ5: Risk-coverage curve")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "rq5_risk_coverage_curve.pdf"), format="pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)

    for frac in [0.1, 0.2, 0.3]:
        keep = int(len(sorted_correct) * (1 - frac))
        if keep == 0:
            continue
        retained_risk = 1 - sorted_correct[:keep].mean()
        print(f"Abstaining on the {int(frac * 100)}% least-confident predictions -> "
              f"error rate on remaining {keep} samples = {retained_risk:.4f} "
              f"(overall error rate = {1 - mc_accuracy:.4f})")


if __name__ == "__main__":
    main()