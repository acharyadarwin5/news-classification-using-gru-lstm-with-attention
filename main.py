"""
=============================================================================
main.py — End-to-End Nepali News Classification Pipeline
=============================================================================
This is the main entry point. It orchestrates the full pipeline:

  1.  Load the dataset from 10 category folders
  2.  Explore data (stats + class distribution plot)
  3.  Preprocess & tokenize Nepali text
  4.  Build Keras Tokenizer + pad sequences + stratified split
  5.  Build & train  BiGRU + Attention  model
  6.  Build & train  BiLSTM + Attention model  (for comparison)
  7.  Build & train  BiGRU + Multi-Head Attention model  (extra)
  8.  Evaluate all three on the held-out test set
  9.  Generate confusion matrices, training curves
  10. Visualize attention weights on sample articles
  11. Plot t-SNE of learned embeddings
  12. Print a GRU vs LSTM comparison table

Run:
    python main.py

GPU acceleration is used automatically if available; the code also works
on CPU (just slower).
=============================================================================
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── TensorFlow setup ─────────────────────────────────────────────────────────
import tensorflow as tf
print(f"TensorFlow version : {tf.__version__}")
print(f"GPU available      : {tf.config.list_physical_devices('GPU')}")

# Allow memory growth so TF doesn't grab all GPU memory at once
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

# ── Project modules ──────────────────────────────────────────────────────────
from config import (
    DATASET_DIR, PLOT_DIR, MODEL_DIR,
    CATEGORY_DISPLAY_NAMES, NUM_CLASSES,
)
from data_utils import (
    load_dataset,
    explore_data,
    preprocess_dataframe,
    build_tokenizer_and_sequences,
    split_data,
    compute_class_weights,
)
from models import (
    build_bigru_attention,
    build_bilstm_attention,
    build_bigru_multihead,
    print_model_summary,
)
from train_eval import (
    compile_and_train,
    evaluate_model,
    plot_training_curves,
    plot_confusion_matrix,
    visualize_attention,
    plot_tsne_embeddings,
)


def main():
    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 1 : Data Loading
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  STEP 1 — Loading Dataset")
    print("═" * 60)
    df = load_dataset(DATASET_DIR)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 1b : Data Exploration
    # ══════════════════════════════════════════════════════════════════════════
    explore_data(df)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 2 : Text Preprocessing
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  STEP 2 — Preprocessing Nepali Text")
    print("═" * 60)
    df = preprocess_dataframe(df)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 2b : Tokenization & Padding
    # ══════════════════════════════════════════════════════════════════════════
    tokenizer, X, y, vocab_size = build_tokenizer_and_sequences(df)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 2c : Train / Val / Test Split (stratified)
    # ══════════════════════════════════════════════════════════════════════════
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # ══════════════════════════════════════════════════════════════════════════
    #  Class Weights (handle any imbalance)
    # ══════════════════════════════════════════════════════════════════════════
    class_weights = compute_class_weights(y_train)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 3 + 4 : Build Models
    # ══════════════════════════════════════════════════════════════════════════
    #
    #  The Embedding layer (Step 3) is the first layer in each model.
    #  It maps integer token IDs → dense float vectors of size EMBED_DIM.
    #  Since Nepali-specific pretrained embeddings are scarce, we train
    #  the embedding from scratch. During back-propagation the model learns
    #  vector representations where semantically similar Nepali words end
    #  up close together in embedding space.
    #
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "═" * 60)
    print("  STEP 3+4 — Building Model Architectures")
    print("═" * 60)

    # --- Model A : BiGRU + Single-Head Attention ---
    gru_train, gru_attn = build_bigru_attention(vocab_size)
    print_model_summary(gru_train)

    # --- Model B : BiLSTM + Single-Head Attention ---
    lstm_train, lstm_attn = build_bilstm_attention(vocab_size)
    print_model_summary(lstm_train)

    # --- Model C : BiGRU + Multi-Head Attention (2 heads) ---
    mh_train, mh_attn = build_bigru_multihead(vocab_size, num_heads=2)
    print_model_summary(mh_train)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 5 : Compile & Train all three models
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "═" * 60)
    print("  STEP 5 — Training Models")
    print("═" * 60)

    # ── Train BiGRU ──
    history_gru = compile_and_train(
        gru_train, X_train, y_train, X_val, y_val,
        class_weight=class_weights, tag="bigru_attention",
    )

    # ── Train BiLSTM ──
    history_lstm = compile_and_train(
        lstm_train, X_train, y_train, X_val, y_val,
        class_weight=class_weights, tag="bilstm_attention",
    )

    # ── Train Multi-Head GRU ──
    history_mh = compile_and_train(
        mh_train, X_train, y_train, X_val, y_val,
        class_weight=class_weights, tag="bigru_multihead",
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 6 : Evaluation
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "═" * 60)
    print("  STEP 6 — Evaluation on Test Set")
    print("═" * 60)

    results = {}

    # -- BiGRU --
    y_pred_gru, acc_gru = evaluate_model(gru_train, X_test, y_test, tag="BiGRU")
    plot_confusion_matrix(y_test, y_pred_gru, tag="BiGRU_Attention")
    plot_training_curves(history_gru, tag="BiGRU_Attention")
    results["BiGRU + Attention"] = acc_gru

    # -- BiLSTM --
    y_pred_lstm, acc_lstm = evaluate_model(lstm_train, X_test, y_test, tag="BiLSTM")
    plot_confusion_matrix(y_test, y_pred_lstm, tag="BiLSTM_Attention")
    plot_training_curves(history_lstm, tag="BiLSTM_Attention")
    results["BiLSTM + Attention"] = acc_lstm

    # -- Multi-Head --
    y_pred_mh, acc_mh = evaluate_model(mh_train, X_test, y_test, tag="MultiHead")
    plot_confusion_matrix(y_test, y_pred_mh, tag="BiGRU_MultiHead")
    plot_training_curves(history_mh, tag="BiGRU_MultiHead")
    results["BiGRU + MultiHead Attn"] = acc_mh

    # ── Comparison Table ──
    print("\n" + "═" * 60)
    print("  📊  MODEL COMPARISON — GRU vs LSTM vs Multi-Head")
    print("═" * 60)
    comp_df = pd.DataFrame([
        {"Model": k, "Test Accuracy": f"{v*100:.2f} %"} for k, v in results.items()
    ])
    print(comp_df.to_string(index=False))
    comp_df.to_csv(os.path.join(PLOT_DIR, "model_comparison.csv"), index=False)
    print(f"\n✅ Comparison table saved → plots/model_comparison.csv")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 7 : Attention Visualization
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "═" * 60)
    print("  STEP 7 — Attention Visualization")
    print("═" * 60)

    # Copy trained weights into the attention models
    # (they share the same graph, so weights are already shared)
    visualize_attention(
        gru_attn, X_test, y_test, tokenizer,
        tag="BiGRU", is_multihead=False,
    )
    visualize_attention(
        mh_attn, X_test, y_test, tokenizer,
        tag="MultiHead", is_multihead=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  EXTRA : t-SNE of Learned Word Embeddings
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "═" * 60)
    print("  EXTRA — t-SNE Embedding Visualization")
    print("═" * 60)

    plot_tsne_embeddings(gru_train, tokenizer, df, tag="BiGRU")

    # ══════════════════════════════════════════════════════════════════════════
    #  Done!
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "🎉" * 30)
    print("  Pipeline complete!")
    print(f"  Models saved in     : {MODEL_DIR}")
    print(f"  Plots & results in  : {PLOT_DIR}")
    print("🎉" * 30 + "\n")


if __name__ == "__main__":
    main()
