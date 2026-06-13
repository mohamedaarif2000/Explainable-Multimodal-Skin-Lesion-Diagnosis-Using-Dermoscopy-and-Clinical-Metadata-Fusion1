import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import BATCH_SIZE, DEVICE, IMG_SIZE, NUM_WORKERS, SEED

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EXCLUDE_COLS = {"patient_id", "lesion_id", "img_id", "img_path", "diagnostic", "label", "biopsed"}


def discover_dataset(data_root):
    """Walk data_root looking for metadata.csv and image files."""
    metadata_path = None
    image_file_index = {}

    for root, _dirs, files in os.walk(data_root):
        for f in files:
            fl = f.lower()
            if fl == "metadata.csv":
                metadata_path = os.path.join(root, f)
            if fl.endswith((".png", ".jpg", ".jpeg")):
                image_file_index[f] = os.path.join(root, f)

    if metadata_path is None:
        raise FileNotFoundError(
            f"Could not find metadata.csv under '{data_root}'. "
            "Download PAD-UFES-20 and update config.DATA_ROOT "
            "(or set the PAD_UFES_20_ROOT environment variable)."
        )

    return metadata_path, image_file_index


class SkinLesionDataset(Dataset):
    def __init__(self, df, meta_df, transform):
        self.paths = df["img_path"].values
        self.labels = df["label"].values
        self.meta = meta_df.values.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        meta = torch.tensor(self.meta[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, meta, label


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, eval_tf


@dataclass
class DataBundle:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_meta: pd.DataFrame
    val_meta: pd.DataFrame
    test_meta: pd.DataFrame
    meta_feature_names: list
    meta_dim: int
    class_names: list
    num_classes: int
    class_weights: torch.Tensor
    train_ds: Dataset
    val_ds: Dataset
    test_ds: Dataset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    eval_tf: transforms.Compose


def load_data(data_root) -> DataBundle:
    metadata_path, image_file_index = discover_dataset(data_root)
    print("Found metadata.csv at:", metadata_path)
    print("Indexed", len(image_file_index), "image files")

    df = pd.read_csv(metadata_path)
    print("Raw metadata shape:", df.shape)

    df["img_path"] = df["img_id"].map(image_file_index)
    n_missing = df["img_path"].isna().sum()
    if n_missing > 0:
        print(f"Dropping {n_missing} rows with no matching image file")
    df = df.dropna(subset=["img_path"]).reset_index(drop=True)
    print("Final dataset size:", df.shape)
    print(df["diagnostic"].value_counts())

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["diagnostic"])
    class_names = list(le.classes_)
    num_classes = len(class_names)
    print("Classes:", class_names)

    meta_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    numeric_cols = [c for c in meta_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in meta_cols if c not in numeric_cols]
    print("Numeric metadata columns:", numeric_cols)
    print("Categorical metadata columns:", categorical_cols)

    for c in categorical_cols:
        df[c] = df[c].astype(str).fillna("UNK").replace({"nan": "UNK"})
    for c in numeric_cols:
        df[c] = df[c].fillna(df[c].median())

    train_df, test_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=SEED)
    train_df, val_df = train_test_split(train_df, test_size=0.1765, stratify=train_df["label"], random_state=SEED)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print("Train:", len(train_df), " Val:", len(val_df), " Test:", len(test_df))

    train_cat = pd.get_dummies(train_df[categorical_cols], columns=categorical_cols)
    val_cat = pd.get_dummies(val_df[categorical_cols], columns=categorical_cols).reindex(columns=train_cat.columns, fill_value=0)
    test_cat = pd.get_dummies(test_df[categorical_cols], columns=categorical_cols).reindex(columns=train_cat.columns, fill_value=0)

    scaler = StandardScaler()
    train_num = pd.DataFrame(scaler.fit_transform(train_df[numeric_cols]), columns=numeric_cols)
    val_num = pd.DataFrame(scaler.transform(val_df[numeric_cols]), columns=numeric_cols)
    test_num = pd.DataFrame(scaler.transform(test_df[numeric_cols]), columns=numeric_cols)

    train_meta = pd.concat([train_num.reset_index(drop=True), train_cat.reset_index(drop=True)], axis=1)
    val_meta = pd.concat([val_num.reset_index(drop=True), val_cat.reset_index(drop=True)], axis=1)
    test_meta = pd.concat([test_num.reset_index(drop=True), test_cat.reset_index(drop=True)], axis=1)

    meta_feature_names = list(train_meta.columns)
    meta_dim = train_meta.shape[1]
    print("Metadata feature dimension:", meta_dim)

    train_tf, eval_tf = build_transforms()
    train_ds = SkinLesionDataset(train_df, train_meta, train_tf)
    val_ds = SkinLesionDataset(val_df, val_meta, eval_tf)
    test_ds = SkinLesionDataset(test_df, test_meta, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Batches -> train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}")

    class_weights_arr = compute_class_weight("balanced", classes=np.arange(num_classes), y=train_df["label"].values)
    class_weights = torch.tensor(class_weights_arr, dtype=torch.float32).to(DEVICE)
    print("Class weights:", dict(zip(class_names, class_weights_arr.round(3))))

    return DataBundle(
        train_df=train_df, val_df=val_df, test_df=test_df,
        train_meta=train_meta, val_meta=val_meta, test_meta=test_meta,
        meta_feature_names=meta_feature_names, meta_dim=meta_dim,
        class_names=class_names, num_classes=num_classes,
        class_weights=class_weights,
        train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        eval_tf=eval_tf,
    )