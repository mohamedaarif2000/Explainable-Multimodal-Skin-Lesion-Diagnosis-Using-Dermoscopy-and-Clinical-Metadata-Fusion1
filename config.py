import os
import random

import numpy as np
import torch


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATA_ROOT = os.environ.get("PAD_UFES_20_ROOT", r"C:\Users\Mohamedaarif\Datasets")

# All RQ scripts write figures/tables/models under outputs/<rq_tag>/
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

EPOCHS = 5          # was 15 — 5 is often enough to see meaningful results
IMG_SIZE = 160      # was 224 — smaller images = much faster conv passes
BATCH_SIZE = 16     # smaller batches reduce per-step memory/CPU pressure on weak machines
NUM_WORKERS = 0

# Model architecture
IMG_EMB_DIM = 256
META_EMB_DIM = 64
SHARED_DIM = 128
N_ATTN_HEADS = 4
DROPOUT = 0.3

# Training
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# RQ2 / RQ4 (SHAP)
SHAP_BACKGROUND_SIZE = 50
SHAP_N_SAMPLES = 100

# RQ5 (Monte Carlo Dropout)
MC_DROPOUT_T = 20


def make_output_dirs(rq_tag):

    output_dir = os.path.join(OUTPUT_ROOT, rq_tag)
    fig_dir = os.path.join(output_dir, "figures")
    tab_dir = os.path.join(output_dir, "tables")
    model_dir = os.path.join(output_dir, "models")
    for d in (output_dir, fig_dir, tab_dir, model_dir):
        os.makedirs(d, exist_ok=True)
    return output_dir, fig_dir, tab_dir, model_dir