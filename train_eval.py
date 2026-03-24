"""
=============================================================================
train_eval.py — Training, Evaluation, and Visualization
=============================================================================
Contains:
  • compile_and_train()        — compiles a model and runs .fit()
  • evaluate_model()           — classification report + test accuracy
  • plot_training_curves()     — accuracy & loss vs epoch
  • plot_confusion_matrix()    — seaborn heatmap of the confusion matrix
  • visualize_attention()      — highlights top-weighted tokens for samples
  • plot_tsne_embeddings()     — t-SNE of learned word vectors coloured by
                                  category of the articles they appear in
=============================================================================
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE

import tensorflow as tf

from config import (
    LEARNING_RATE, BATCH_SIZE, EPOCHS,
    EARLY_STOP_PATIENCE, LR_REDUCE_PATIENCE, LR_REDUCE_FACTOR,
    MODEL_DIR, PLOT_DIR,
    NUM_CLASSES, CATEGORY_DISPLAY_NAMES,
    NUM_ATTENTION_SAMPLES, TSNE_SAMPLE_SIZE, MAX_LEN,
)

# ─────────────────────────── Matplotlib defaults ───────────────────────────
try:
    matplotlib.rcParams["font.family"] = "Noto Sans Devanagari"
except Exception:
    pass
sns.set_style("whitegrid")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5 : Compile & Train
# ═══════════════════════════════════════════════════════════════════════════════

def compile_and_train(
    model,
    X_train, y_train,
    X_val, y_val,
    class_weight: dict = None,
    tag: str = "model",
):
    """
    Compile the model with Adam + sparse_categorical_crossentropy and train it
    with EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks.

    Parameters
    ----------
    model       : tf.keras.Model  (single-output training model)
    tag         : str  — used to name checkpoint files & plots

    Returns
    -------
    history : tf.keras.callbacks.History
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_path = os.path.join(MODEL_DIR, f"best_{tag}.keras")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            verbose=1,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print(f"\n🚀 Training  {model.name}  (batch={BATCH_SIZE}, max epochs={EPOCHS})")
    print("─" * 60)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\n✅ Best model saved → {checkpoint_path}")
    return history


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 6 : Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test, tag="model"):
    """
    Print classification report, overall accuracy, and return predictions.
    """
    print(f"\n{'=' * 60}")
    print(f"  📈  Evaluation — {model.name}")
    print(f"{'=' * 60}")

    loss, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)
    print(f"\n  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {acc:.4f}  ({acc*100:.2f} %)")

    y_pred = np.argmax(model.predict(X_test, batch_size=BATCH_SIZE), axis=-1)

    print("\n▸ Classification Report:\n")
    print(classification_report(
        y_test, y_pred,
        target_names=CATEGORY_DISPLAY_NAMES,
        digits=4,
    ))

    return y_pred, acc


def plot_confusion_matrix(y_test, y_pred, tag="model"):
    """Seaborn heatmap of the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CATEGORY_DISPLAY_NAMES,
        yticklabels=CATEGORY_DISPLAY_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {tag}", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"confusion_matrix_{tag}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ Confusion matrix saved → {path}")


def plot_training_curves(history, tag="model"):
    """Plot accuracy and loss curves for training vs validation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Accuracy ---
    axes[0].plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    axes[0].set_title(f"Accuracy — {tag}", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Loss ---
    axes[1].plot(history.history["loss"], label="Train Loss", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    axes[1].set_title(f"Loss — {tag}", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"training_curves_{tag}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ Training curves saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 7 : Attention Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def _id_to_word(tokenizer):
    """Return dict  {token_id: word}."""
    return {v: k for k, v in tokenizer.word_index.items()}


def visualize_attention(
    attention_model,
    X_test, y_test,
    tokenizer,
    num_samples: int = NUM_ATTENTION_SAMPLES,
    tag: str = "model",
    is_multihead: bool = False,
):
    """
    For a handful of test samples:
      1. Run the attention_model (multi-output) to get predictions + weights
      2. Map token IDs back to Nepali words
      3. Print the words coloured / scored by attention
      4. Save a bar-chart of attention weights for each sample
    """
    id2word = _id_to_word(tokenizer)
    indices = np.random.choice(len(X_test), size=num_samples, replace=False)

    print(f"\n{'=' * 60}")
    print(f"  🔍  Attention Visualization — {tag}")
    print(f"{'=' * 60}")

    for idx in indices:
        x_sample = X_test[idx: idx + 1]
        true_label = int(y_test[idx])

        preds, attn_w = attention_model.predict(x_sample, verbose=0)
        pred_label = int(np.argmax(preds, axis=-1)[0])

        # For multi-head attention, average across heads for visualisation
        if is_multihead and attn_w.ndim == 3:
            attn_w = np.mean(attn_w, axis=1)  # (1, T)

        attn_w = attn_w[0]  # (T,)
        token_ids = x_sample[0]  # (MAX_LEN,)

        # Build list of (word, weight) only for non-padding tokens
        word_weights = []
        for t_id, w in zip(token_ids, attn_w):
            if t_id == 0:  # padding
                continue
            word = id2word.get(t_id, "<UNK>")
            word_weights.append((word, float(w)))

        # Normalise weights for display
        max_w = max(w for _, w in word_weights) if word_weights else 1
        norm_ww = [(wd, w / max_w) for wd, w in word_weights]

        print(f"\n{'─' * 50}")
        print(f"  True: {CATEGORY_DISPLAY_NAMES[true_label]}  |  "
              f"Pred: {CATEGORY_DISPLAY_NAMES[pred_label]}")
        print(f"{'─' * 50}")

        # Print top 20 attended words
        sorted_ww = sorted(word_weights, key=lambda x: x[1], reverse=True)
        print("  Top-20 attended tokens:")
        for rank, (wd, w) in enumerate(sorted_ww[:20], 1):
            bar = "█" * int(w / max_w * 30)
            print(f"    {rank:2d}. {wd:20s}  {w:.6f}  {bar}")

        # ---- Save bar chart of attention over sequence ----
        if len(word_weights) > 0:
            words_plot = [w for w, _ in word_weights[:80]]  # first 80 tokens
            weights_plot = [w for _, w in word_weights[:80]]

            fig, ax = plt.subplots(figsize=(16, 4))
            ax.bar(range(len(weights_plot)), weights_plot, color="#5e81ac", width=0.8)
            ax.set_xticks(range(len(words_plot)))
            ax.set_xticklabels(words_plot, rotation=90, fontsize=6)
            ax.set_title(
                f"Attention Weights — True: {CATEGORY_DISPLAY_NAMES[true_label]} "
                f"| Pred: {CATEGORY_DISPLAY_NAMES[pred_label]}",
                fontsize=11,
            )
            ax.set_ylabel("Attention Weight")
            plt.tight_layout()
            path = os.path.join(PLOT_DIR, f"attention_{tag}_sample{idx}.png")
            plt.savefig(path, dpi=150)
            plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  EXTRA : t-SNE of Learned Embeddings
# ═══════════════════════════════════════════════════════════════════════════════

def plot_tsne_embeddings(
    model,
    tokenizer,
    df,
    tag: str = "model",
    sample_size: int = TSNE_SAMPLE_SIZE,
):
    """
    Extract the learned embedding matrix, subsample tokens, project to 2-D
    with t-SNE, and colour dots by the dominant category each word appears in.
    """
    print(f"\n⏳ Computing t-SNE for learned embeddings ({tag}) …")

    # Get embedding weights
    emb_layer = model.get_layer("embedding")
    emb_matrix = emb_layer.get_weights()[0]  # (vocab_size, EMBED_DIM)

    word2id = tokenizer.word_index
    id2word = {v: k for k, v in word2id.items()}

    # Build a mapping: word → dominant category
    from collections import Counter, defaultdict
    word_cat_counter = defaultdict(Counter)
    for _, row in df.iterrows():
        for token in str(row["text_clean"]).split():
            if token in word2id:
                word_cat_counter[token][row["label"]] += 1

    word_dominant_cat = {}
    for word, counter in word_cat_counter.items():
        word_dominant_cat[word] = counter.most_common(1)[0][0]

    # Subsample
    all_ids = list(range(1, min(len(emb_matrix), sample_size)))
    vectors = emb_matrix[all_ids]
    labels = []
    for idx in all_ids:
        word = id2word.get(idx, "")
        labels.append(word_dominant_cat.get(word, -1))

    labels = np.array(labels)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    reduced = tsne.fit_transform(vectors)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.cm.get_cmap("tab10", NUM_CLASSES)

    for cat_idx in range(NUM_CLASSES):
        mask = labels == cat_idx
        if mask.sum() == 0:
            continue
        ax.scatter(
            reduced[mask, 0], reduced[mask, 1],
            s=8, alpha=0.5, label=CATEGORY_DISPLAY_NAMES[cat_idx],
            color=cmap(cat_idx),
        )

    ax.set_title(f"t-SNE of Learned Word Embeddings — {tag}", fontsize=14)
    ax.legend(markerscale=4, fontsize=9, loc="best")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"tsne_embeddings_{tag}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ t-SNE plot saved → {path}")
