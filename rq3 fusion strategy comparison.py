"""
RQ3 - Does attention-based multimodal fusion outperform traditional feature
concatenation methods for skin lesion classification?

Evaluation: Compare Early Fusion, Late Fusion, and Attention Fusion on
Accuracy, Precision, Recall, F1-Score, ROC-AUC and per-class F1.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import f1_score

from config import DATA_ROOT, EPOCHS, make_output_dirs
from data_pipeline import load_data
from engine import evaluate_model, train_model
from models import AttentionFusionModel, EarlyFusionModel, LateFusionModel
from plotting import plot_confusion_matrix, plot_metric_comparison, plot_roc_curves, plot_training_curves

RQ_TAG = "rq3_fusion_strategy_comparison"
METRIC_COLS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro"]


def main():
    output_dir, fig_dir, tab_dir, model_dir = make_output_dirs(RQ_TAG)
    print("Outputs will be saved under:", output_dir)

    data = load_data(DATA_ROOT)

    print("\nTraining early fusion model...")
    early_model = EarlyFusionModel(data.meta_dim, data.num_classes)
    early_model, hist_early = train_model(
        early_model, data.train_loader, data.val_loader, data.class_weights, model_dir,
        epochs=EPOCHS, modality="multimodal", tag="early_fusion",
    )

    print("\nTraining late fusion model...")
    late_model = LateFusionModel(data.meta_dim, data.num_classes)
    late_model, hist_late = train_model(
        late_model, data.train_loader, data.val_loader, data.class_weights, model_dir,
        epochs=EPOCHS, modality="multimodal", tag="late_fusion",
    )

    print("\nTraining attention fusion model (proposed)...")
    attn_model = AttentionFusionModel(data.meta_dim, data.num_classes)
    attn_model, hist_attn = train_model(
        attn_model, data.train_loader, data.val_loader, data.class_weights, model_dir,
        epochs=EPOCHS, modality="multimodal", tag="attention_fusion_rq3",
    )

    plot_training_curves(hist_early, "Early fusion: training curves", "rq3_training_curves_early_fusion.pdf", fig_dir)
    plot_training_curves(hist_late, "Late fusion: training curves", "rq3_training_curves_late_fusion.pdf", fig_dir)
    plot_training_curves(hist_attn, "Attention fusion: training curves", "rq3_training_curves_attention_fusion.pdf", fig_dir)

    late_weight = torch.sigmoid(late_model.fusion_weight).item()
    print(f"Learned late-fusion weight (image vs. metadata): "
          f"w_image={late_weight:.3f}, w_meta={1 - late_weight:.3f}")

    results = []
    per_class_f1 = {}

    for name, model in [("Early Fusion", early_model),
                         ("Late Fusion", late_model),
                         ("Attention Fusion", attn_model)]:
        metrics, labels, preds, probs = evaluate_model(model, data.test_loader, modality="multimodal")
        metrics["model"] = name
        results.append(metrics)

        per_class_f1[name] = f1_score(labels, preds, average=None, zero_division=0)

        safe_name = name.lower().replace(" ", "_")
        plot_confusion_matrix(labels, preds, data.class_names,
                               f"{name}: confusion matrix (test set)",
                               f"rq3_confusion_{safe_name}.pdf", fig_dir)
        plot_roc_curves(labels, probs, data.class_names,
                         f"{name}: ROC curves (test set)",
                         f"rq3_roc_{safe_name}.pdf", fig_dir)

        print(name, "->", {k: round(v, 4) for k, v in metrics.items() if k != "model"})

    results_df = pd.DataFrame(results)[["model"] + METRIC_COLS + ["balanced_accuracy"]]
    print(results_df)

    per_class_df = pd.DataFrame(per_class_f1, index=data.class_names).T
    per_class_df.index.name = "model"
    per_class_df = per_class_df.reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    per_class_df.set_index("model")[data.class_names].plot(kind="bar", ax=ax)
    ax.set_ylabel("F1 score")
    ax.set_title("RQ3: Per-class F1 by fusion strategy (test set)")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "rq3_per_class_f1.pdf"), format="pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(per_class_df)

    results_path = os.path.join(tab_dir, "rq3_results.csv")
    results_df.to_csv(results_path, index=False)
    print("Saved", results_path)

    per_class_path = os.path.join(tab_dir, "rq3_per_class_f1.csv")
    per_class_df.to_csv(per_class_path, index=False)
    print("Saved", per_class_path)

    plot_metric_comparison(results_df, METRIC_COLS,
                            "RQ3: Early vs Late vs Attention Fusion (test set)",
                            "rq3_metric_comparison.pdf", fig_dir)


if __name__ == "__main__":
    main()