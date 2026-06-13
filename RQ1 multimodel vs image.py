"""
RQ1 - How much does multimodal fusion of dermoscopy images and clinical metadata
improve skin lesion classification compared to image-only models?

Evaluation: Compare Image-only vs Multimodal (Attention Fusion) models on
Accuracy, Precision, Recall, F1-Score and ROC-AUC.
"""

import os

import pandas as pd

from config import DATA_ROOT, EPOCHS, make_output_dirs
from data_pipeline import load_data
from engine import evaluate_model, train_model
from models import AttentionFusionModel, ImageOnlyModel
from plotting import plot_confusion_matrix, plot_metric_comparison, plot_roc_curves, plot_training_curves

RQ_TAG = "rq1_multimodal_vs_image_only"
METRIC_COLS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro"]


def main():
    output_dir, fig_dir, tab_dir, model_dir = make_output_dirs(RQ_TAG)
    print("Outputs will be saved under:", output_dir)

    data = load_data(DATA_ROOT)

    print("\nTraining image-only model (EfficientNet-B3)...")
    image_model = ImageOnlyModel(data.num_classes)
    image_model, hist_image = train_model(
        image_model, data.train_loader, data.val_loader, data.class_weights, model_dir,
        epochs=EPOCHS, modality="image", tag="image_only",
    )

    print("\nTraining multimodal attention-fusion model (proposed method)...")
    fusion_model = AttentionFusionModel(data.meta_dim, data.num_classes)
    fusion_model, hist_fusion = train_model(
        fusion_model, data.train_loader, data.val_loader, data.class_weights, model_dir,
        epochs=EPOCHS, modality="multimodal", tag="attention_fusion",
    )

    plot_training_curves(hist_image, "Image-only model: training curves",
                          "rq1_training_curves_image_only.pdf", fig_dir)
    plot_training_curves(hist_fusion, "Multimodal attention-fusion model: training curves",
                          "rq1_training_curves_attention_fusion.pdf", fig_dir)

    results = []
    for name, model, modality in [("Image-only", image_model, "image"),
                                   ("Multimodal (Attention Fusion)", fusion_model, "multimodal")]:
        metrics, labels, preds, probs = evaluate_model(model, data.test_loader, modality=modality)
        metrics["model"] = name
        results.append(metrics)

        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        plot_confusion_matrix(labels, preds, data.class_names,
                               f"{name}: confusion matrix (test set)",
                               f"rq1_confusion_{safe_name}.pdf", fig_dir)
        plot_roc_curves(labels, probs, data.class_names,
                         f"{name}: ROC curves (test set)",
                         f"rq1_roc_{safe_name}.pdf", fig_dir)

        print(name, "->", {k: round(v, 4) for k, v in metrics.items() if k != "model"})

    results_df = pd.DataFrame(results)[["model"] + METRIC_COLS + ["balanced_accuracy"]]
    print(results_df)

    results_path = os.path.join(tab_dir, "rq1_results.csv")
    results_df.to_csv(results_path, index=False)
    print("Saved", results_path)

    plot_metric_comparison(results_df, METRIC_COLS,
                            "RQ1: Image-only vs Multimodal (Attention Fusion) - test set",
                            "rq1_metric_comparison.pdf", fig_dir)

    improvement = (results_df.set_index("model").loc["Multimodal (Attention Fusion)", METRIC_COLS]
                   - results_df.set_index("model").loc["Image-only", METRIC_COLS])
    improvement_df = improvement.reset_index()
    improvement_df.columns = ["metric", "absolute_improvement"]
    improvement_path = os.path.join(tab_dir, "rq1_improvement.csv")
    improvement_df.to_csv(improvement_path, index=False)
    print("Saved", improvement_path)
    print(improvement_df)


if __name__ == "__main__":
    main()
