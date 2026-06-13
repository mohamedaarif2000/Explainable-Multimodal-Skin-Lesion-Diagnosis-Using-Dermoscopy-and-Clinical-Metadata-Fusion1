"""
Generic training and evaluation loops used by all RQ scripts.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                              precision_score, recall_score, roc_auc_score)

from config import DEVICE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY


def forward_by_modality(model, img, meta, modality):
    if modality == "image":
        return model(img)
    elif modality == "meta":
        return model(meta)
    else:
        return model(img, meta)


def train_model(model, train_loader, val_loader, class_weights, model_dir,
                 epochs=EPOCHS, lr=LEARNING_RATE, modality="multimodal", tag="model"):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_bacc": []}
    best_val_bacc = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for img, meta, label in train_loader:
            img, meta, label = img.to(DEVICE), meta.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            out = forward_by_modality(model, img, meta, modality)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * img.size(0)
        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for img, meta, label in val_loader:
                img, meta, label = img.to(DEVICE), meta.to(DEVICE), label.to(DEVICE)
                out = forward_by_modality(model, img, meta, modality)
                loss = criterion(out, label)
                val_loss += loss.item() * img.size(0)
                all_preds.append(out.argmax(1).cpu())
                all_labels.append(label.cpu())
        val_loss /= len(val_loader.dataset)
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        val_acc = accuracy_score(all_labels, all_preds)
        val_bacc = balanced_accuracy_score(all_labels, all_preds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_bacc"].append(val_bacc)

        print(f"[{tag}] epoch {epoch + 1:02d}/{epochs} | "
              f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
              f"val_acc {val_acc:.4f} | val_bacc {val_bacc:.4f}")

        if val_bacc > best_val_bacc:
            best_val_bacc = val_bacc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(model_dir, f"{tag}_best.pt"))
    return model, history


def evaluate_model(model, loader, modality="multimodal"):
    """Returns accuracy, precision, recall, F1 (macro), balanced accuracy and macro-AUC."""
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for img, meta, label in loader:
            img, meta = img.to(DEVICE), meta.to(DEVICE)
            out = forward_by_modality(model, img, meta, modality)
            probs = F.softmax(out, dim=1).cpu().numpy()
            probs_all.append(probs)
            labels_all.append(label.numpy())
    probs_all = np.concatenate(probs_all)
    labels_all = np.concatenate(labels_all)
    preds_all = probs_all.argmax(1)

    metrics = {
        "accuracy": accuracy_score(labels_all, preds_all),
        "balanced_accuracy": balanced_accuracy_score(labels_all, preds_all),
        "precision_macro": precision_score(labels_all, preds_all, average="macro", zero_division=0),
        "recall_macro": recall_score(labels_all, preds_all, average="macro", zero_division=0),
        "f1_macro": f1_score(labels_all, preds_all, average="macro", zero_division=0),
    }
    try:
        metrics["roc_auc_macro"] = roc_auc_score(labels_all, probs_all, multi_class="ovr", average="macro")
    except Exception:
        metrics["roc_auc_macro"] = float("nan")

    return metrics, labels_all, preds_all, probs_all