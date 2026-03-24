"""
=============================================================================
data_utils.py — Data Loading, Exploration, and Preprocessing
=============================================================================
Handles:
  • Walking the 10 category folders and loading every .txt file
  • Cleaning raw Nepali (Devanagari) text
  • Building a Keras Tokenizer vocabulary
  • Padding sequences to a fixed length
  • Stratified train / validation / test splits
=============================================================================
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter

# Use a font that can render Devanagari if available, fall back gracefully
matplotlib.rcParams["font.family"] = "Noto Sans Devanagari"

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import (
    DATASET_DIR, CATEGORY_FOLDERS, CATEGORY_DISPLAY_NAMES,
    MAX_VOCAB_SIZE, MAX_LEN, OOV_TOKEN,
    TEST_SIZE, VAL_SIZE, RANDOM_STATE, PLOT_DIR,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Basic Nepali Stopwords
# ─────────────────────────────────────────────────────────────────────────────
NEPALI_STOPWORDS = set([
    "छ", "छन्", "छु", "छौ", "छिन्", "छे", "छन",
    "हो", "हुन्", "हुन्छ", "भएको", "भएका", "गरेको", "गरेका",
    "गर्न", "गर्ने", "गरी", "गरिएको",
    "को", "का", "की", "कि",
    "मा", "ले", "लाई", "बाट", "देखि", "सम्म", "सँग", "संग",
    "र", "तथा", "वा", "अथवा", "कि",
    "यो", "यी", "ती", "त्यो", "उ", "उनी", "उनीहरू",
    "म", "हामी", "तिमी", "तपाईं", "तपाईँ",
    "यस", "यसको", "त्यस", "त्यसको",
    "एक", "दुई", "तीन",
    "पनि", "नै", "त", "न", "हैन",
    "थियो", "थिए", "थिइन्",
    "भने", "अनि", "तर", "यदि", "किनभने",
    "छैन", "छैनन्",
    "सो", "जुन", "जो", "जसले", "जसको",
    "कुनै", "केही", "सबै", "अरू", "अन्य",
    "हुने", "रहेको", "रहेका", "भएर",
    "अब", "अहिले", "यहाँ", "त्यहाँ",
    "गर्दा", "गर्दै", "गरे", "गरिन",
    "दिएको", "दिएका", "दिने", "दिन",
    "अनुसार", "बारे", "बारेमा", "प्रति",
    "नि", "चाहिँ", "भन्ने", "भन्दा",
    "जस्तो", "जस्ता", "जस्तै",
    "लागि", "निम्ति",
    "हुँदा", "हुँदै", "भइ",
    "आफ्नो", "आफू", "आफ्ना",
    "पर्छ", "पर्ने", "पर्दछ",
    "सक्छ", "सक्ने", "सकेको",
    "रूपमा", "रूपले",
    "मात्र", "मात्रै",
])


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 : Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(dataset_dir: str = DATASET_DIR) -> pd.DataFrame:
    """
    Walk every category folder, read every .txt file, and build a DataFrame.

    Returns
    -------
    pd.DataFrame with columns  ['text', 'label', 'category_name']
        label is an integer 0–9 matching the index in CATEGORY_FOLDERS.
    """
    records = []
    for label_idx, folder_name in enumerate(CATEGORY_FOLDERS):
        folder_path = os.path.join(dataset_dir, folder_name)
        if not os.path.isdir(folder_path):
            print(f"⚠  Folder not found, skipping: {folder_path}")
            continue

        display_name = CATEGORY_DISPLAY_NAMES[label_idx]
        file_count = 0

        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            if not os.path.isfile(fpath):
                continue
            # Try multiple encodings common for Nepali text
            text = None
            for enc in ("utf-8", "utf-8-sig", "utf-16", "latin-1"):
                try:
                    with open(fpath, "r", encoding=enc) as f:
                        text = f.read()
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue

            if text is None or len(text.strip()) == 0:
                continue

            records.append({
                "text": text.strip(),
                "label": label_idx,
                "category_name": display_name,
            })
            file_count += 1

        print(f"  ✔ {display_name:20s} — loaded {file_count:,} articles")

    df = pd.DataFrame(records)
    print(f"\n📦 Total articles loaded: {len(df):,}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1b : Data Exploration
# ═══════════════════════════════════════════════════════════════════════════════

def explore_data(df: pd.DataFrame) -> None:
    """Print summary statistics and plot class distribution."""

    print("\n" + "=" * 60)
    print("  📊  DATA EXPLORATION")
    print("=" * 60)

    # Class distribution
    dist = df["category_name"].value_counts()
    print("\n▸ Class distribution:")
    print(dist.to_string())

    # Article lengths (in characters and words)
    df["char_len"] = df["text"].apply(len)
    df["word_count"] = df["text"].apply(lambda t: len(t.split()))

    print(f"\n▸ Average article length : {df['char_len'].mean():,.0f} chars "
          f"/ {df['word_count'].mean():,.0f} words")
    print(f"▸ Median article length  : {df['char_len'].median():,.0f} chars "
          f"/ {df['word_count'].median():,.0f} words")
    print(f"▸ Max article length     : {df['char_len'].max():,} chars "
          f"/ {df['word_count'].max():,} words")

    # Rough vocabulary size (whitespace tokenised, before cleaning)
    all_tokens = Counter()
    for text in df["text"]:
        all_tokens.update(text.split())
    print(f"▸ Unique tokens (raw)    : {len(all_tokens):,}")

    # Plot class distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    dist.plot(kind="bar", ax=ax, color="#5e81ac", edgecolor="#2e3440")
    ax.set_title("Class Distribution — Nepali News Categories", fontsize=14)
    ax.set_ylabel("Number of Articles")
    ax.set_xlabel("Category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "class_distribution.png"), dpi=150)
    plt.close()
    print(f"\n✅ Class distribution plot saved → plots/class_distribution.png")

    # Clean up temp cols
    df.drop(columns=["char_len", "word_count"], inplace=True, errors="ignore")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 : Text Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Clean a single Nepali article:
      1. Remove HTML tags / artifacts
      2. Remove URLs
      3. Remove English letters & digits
      4. Remove punctuation & special characters (keep Devanagari + spaces)
      5. Normalise whitespace
      6. Optionally remove Nepali stopwords
    """
    # 1. Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Remove English letters and digits (ASCII 0-9, a-z, A-Z)
    text = re.sub(r"[a-zA-Z0-9]", " ", text)

    # 4. Keep only Devanagari characters (Unicode block 0900–097F) and spaces
    text = re.sub(r"[^\u0900-\u097F\s]", " ", text)

    # 5. Collapse multiple spaces / newlines
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Remove stopwords
    if remove_stopwords:
        tokens = text.split()
        tokens = [t for t in tokens if t not in NEPALI_STOPWORDS]
        text = " ".join(tokens)

    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clean_text to every row and drop empties."""
    print("\n⏳ Cleaning Nepali text (this may take a moment) …")
    df = df.copy()
    df["text_clean"] = df["text"].apply(clean_text)
    # Drop any rows that became empty after cleaning
    before = len(df)
    df = df[df["text_clean"].str.strip().astype(bool)].reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"  ⚠ Dropped {before - after} empty articles after cleaning")
    print("  ✔ Text cleaning complete")
    return df


def build_tokenizer_and_sequences(df: pd.DataFrame):
    """
    Build a Keras Tokenizer on the cleaned text, convert texts to integer
    sequences, and pad them to MAX_LEN.

    Returns
    -------
    tokenizer : tf.keras.preprocessing.text.Tokenizer
    X_padded  : np.ndarray  shape (N, MAX_LEN)
    y         : np.ndarray  shape (N,)
    """
    tokenizer = Tokenizer(
        num_words=MAX_VOCAB_SIZE,
        oov_token=OOV_TOKEN,
        filters="",          # we already cleaned — no extra filtering
    )
    tokenizer.fit_on_texts(df["text_clean"])
    sequences = tokenizer.texts_to_sequences(df["text_clean"])

    X_padded = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )
    y = df["label"].values

    vocab_size = min(MAX_VOCAB_SIZE, len(tokenizer.word_index)) + 1  # +1 for pad
    print(f"\n📖 Tokenizer vocabulary : {len(tokenizer.word_index):,} unique tokens")
    print(f"   Using top {vocab_size:,} tokens (incl. padding & OOV)")
    print(f"   Sequence length      : {MAX_LEN}")
    print(f"   X shape              : {X_padded.shape}")
    print(f"   y shape              : {y.shape}")

    return tokenizer, X_padded, y, vocab_size


def split_data(X, y):
    """
    Stratified split into train (70 %), validation (15 %), test (15 %).

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: 85 % train+val  vs  15 % test
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    # Second split: from the 85 %, take ~17.6 % as val → 15 % of total
    val_fraction = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_fraction, random_state=RANDOM_STATE, stratify=y_tv,
    )
    print(f"\n🔀 Data splits (stratified):")
    print(f"   Train : {X_train.shape[0]:,}  ({X_train.shape[0]/len(X)*100:.1f} %)")
    print(f"   Val   : {X_val.shape[0]:,}  ({X_val.shape[0]/len(X)*100:.1f} %)")
    print(f"   Test  : {X_test.shape[0]:,}  ({X_test.shape[0]/len(X)*100:.1f} %)")
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_class_weights(y_train):
    """
    Compute class weights to handle any imbalance.
    Returns a dict  {class_idx: weight}  usable by model.fit().
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    print("\n⚖  Class weights:")
    for c, w in class_weight_dict.items():
        print(f"   Class {c}: {w:.4f}")
    return class_weight_dict
