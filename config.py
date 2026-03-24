"""
=============================================================================
config.py — Central Configuration for Nepali News Classification System
=============================================================================
All hyperparameters, file paths, and category mappings are defined here.
Change this file to tune the model without modifying any other source files.
=============================================================================
"""

import os

# ─────────────────────────── Paths ───────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

# Create output directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ─────────────────────────── Category Mapping ───────────────────────────
# Folder names in the dataset → human-readable display names
CATEGORY_FOLDERS = [
    "ArthaBanijya",
    "Bichar",
    "Desh",
    "Khelkud",
    "Manoranjan",
    "Prabas",
    "Sahitya",
    "SuchanaPrabidhi",
    "Swasthya",
    "Viswa",
]

CATEGORY_DISPLAY_NAMES = [
    "ArthaBanijya",
    "Bichar",
    "Desh",
    "Khelkud",
    "Manoranjan",
    "Prabas",
    "Sahitya",
    "SuchanaPrabidhi",
    "Swasthya",
    "Viswa",
]

NUM_CLASSES = len(CATEGORY_FOLDERS)   # 10

# ─────────────────────────── Text / Tokenizer ───────────────────────────
MAX_VOCAB_SIZE = 30_000    # Keep the top-30k most frequent tokens
MAX_LEN = 300              # Pad / truncate every article to 300 tokens
OOV_TOKEN = "<OOV>"        # Out-of-vocabulary placeholder

# ─────────────────────────── Embedding ───────────────────────────
EMBED_DIM = 128            # Dimensionality of learned word vectors

# ─────────────────────────── Model Architecture ───────────────────────────
GRU_UNITS_1 = 128          # First Bidirectional GRU layer units
GRU_UNITS_2 = 64           # Second Bidirectional GRU layer units
SPATIAL_DROPOUT = 0.3      # SpatialDropout1D rate after embedding
DROPOUT_1 = 0.4            # Dropout after attention context vector
DENSE_UNITS = 64           # Dense layer units before final output
DROPOUT_2 = 0.3            # Dropout after dense layer

# ─────────────────────────── Training ───────────────────────────
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 30
EARLY_STOP_PATIENCE = 5
LR_REDUCE_PATIENCE = 2
LR_REDUCE_FACTOR = 0.5

# ─────────────────────────── Data Splits ───────────────────────────
TEST_SIZE = 0.15           # 15 % held out for final test
VAL_SIZE = 0.15            # 15 % of total for validation  →  ~17.6 % of remaining
RANDOM_STATE = 42

# ─────────────────────────── Visualization ───────────────────────────
NUM_ATTENTION_SAMPLES = 5  # How many test articles to visualize attention for
TSNE_SAMPLE_SIZE = 3000    # Subsample for t-SNE (full vocab is slow)
