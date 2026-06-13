"""
Plotting helpers shared by all RQ scripts. All figures are saved as PDF.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import confusion_matrix, roc_curve


def plot_confusion_matrix(labels, preds, class_names, title, filename, fig_dir):
    cm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, filename), format="pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_training_curves(history, title, filename, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train loss")
    axes[0].plot(history["val_loss"], label="val loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].set_title("Loss")

    axes[1].plot(history["val_acc"], label="val accuracy")
    axes[1].plot(history["val_bacc"], label="val balanced accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score"); axes[1].legend(); axes[1].set_title("Validation accuracy")

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, filename), format="pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_roc_curves(labels, probs, class_names, title, filename, fig_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    labels_bin = np.eye(len(class_names))[labels]
    for i, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc_val = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{cname} (AUC={roc_auc_val:.2f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, filename), format="pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_metric_comparison(results_df, metric_cols, title, filename, fig_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    results_df.set_index("model")[metric_cols].plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, filename), format="pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)