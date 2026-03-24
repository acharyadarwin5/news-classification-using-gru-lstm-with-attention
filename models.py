"""
=============================================================================
models.py — Model Architectures for Nepali News Classification
=============================================================================
Provides factory functions that return *two* Keras Models each:
  1. training_model  — standard single-output model  (logits only)
  2. attention_model — multi-output model sharing the same weights
                        (logits + attention weights)  for visualisation

Architectures implemented:
  • Bidirectional GRU  + single-head attention
  • Bidirectional LSTM + single-head attention
  • Bidirectional GRU  + multi-head attention (2 heads)
=============================================================================
"""

import tensorflow as tf
from attention import AttentionLayer, MultiHeadAttentionLayer
from config import (
    MAX_LEN, EMBED_DIM, NUM_CLASSES,
    GRU_UNITS_1, GRU_UNITS_2,
    SPATIAL_DROPOUT, DROPOUT_1, DENSE_UNITS, DROPOUT_2,
)


def _common_head(context, name_prefix=""):
    """Shared classification head after the attention context vector."""
    x = tf.keras.layers.Dropout(DROPOUT_1, name=f"{name_prefix}dropout_1")(context)
    x = tf.keras.layers.Dense(DENSE_UNITS, activation="relu",
                               name=f"{name_prefix}dense_hidden")(x)
    x = tf.keras.layers.Dropout(DROPOUT_2, name=f"{name_prefix}dropout_2")(x)
    output = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax",
                                    name=f"{name_prefix}output")(x)
    return output


# ═══════════════════════════════════════════════════════════════════════════════
#  Bidirectional GRU + Single-Head Attention
# ═══════════════════════════════════════════════════════════════════════════════

def build_bigru_attention(vocab_size: int):
    """
    Architecture
    ------------
    Input → Embedding → SpatialDropout1D
      → BiGRU(128, return_seq) → BiGRU(64, return_seq)
      → Attention → Dropout → Dense(64, relu) → Dropout → Softmax(10)

    Returns
    -------
    train_model     : Model(inputs, outputs=logits)
    attention_model : Model(inputs, outputs=[logits, attention_weights])
    """
    inp = tf.keras.layers.Input(shape=(MAX_LEN,), name="token_ids")

    # --- Embedding ---
    # Maps each integer token ID to a dense float vector of size EMBED_DIM.
    # These vectors are learned from scratch during training so the model
    # discovers meaningful representations for Nepali words.
    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=EMBED_DIM,
        input_length=MAX_LEN,
        name="embedding",
    )(inp)

    x = tf.keras.layers.SpatialDropout1D(SPATIAL_DROPOUT, name="spatial_drop")(x)

    # --- Stacked Bidirectional GRU ---
    # Layer 1: 128 units → output dim = 256 (128 fwd + 128 bwd)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(GRU_UNITS_1, return_sequences=True, name="gru_1"),
        name="bigru_1",
    )(x)

    # Layer 2: 64 units → output dim = 128 (64 fwd + 64 bwd)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(GRU_UNITS_2, return_sequences=True, name="gru_2"),
        name="bigru_2",
    )(x)

    # --- Attention ---
    context, attn_weights = AttentionLayer(name="attention")(x)

    # --- Classifier head ---
    logits = _common_head(context, name_prefix="gru_")

    train_model = tf.keras.Model(inputs=inp, outputs=logits,
                                  name="BiGRU_Attention")
    attention_model = tf.keras.Model(inputs=inp,
                                      outputs=[logits, attn_weights],
                                      name="BiGRU_Attention_viz")

    return train_model, attention_model


# ═══════════════════════════════════════════════════════════════════════════════
#  Bidirectional LSTM + Single-Head Attention
# ═══════════════════════════════════════════════════════════════════════════════

def build_bilstm_attention(vocab_size: int):
    """
    Same architecture as BiGRU but replaces GRU cells with LSTM cells.
    Allows comparison of GRU vs LSTM performance.
    """
    inp = tf.keras.layers.Input(shape=(MAX_LEN,), name="token_ids")

    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=EMBED_DIM,
        input_length=MAX_LEN,
        name="embedding",
    )(inp)

    x = tf.keras.layers.SpatialDropout1D(SPATIAL_DROPOUT, name="spatial_drop")(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(GRU_UNITS_1, return_sequences=True, name="lstm_1"),
        name="bilstm_1",
    )(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(GRU_UNITS_2, return_sequences=True, name="lstm_2"),
        name="bilstm_2",
    )(x)

    context, attn_weights = AttentionLayer(name="attention")(x)

    logits = _common_head(context, name_prefix="lstm_")

    train_model = tf.keras.Model(inputs=inp, outputs=logits,
                                  name="BiLSTM_Attention")
    attention_model = tf.keras.Model(inputs=inp,
                                      outputs=[logits, attn_weights],
                                      name="BiLSTM_Attention_viz")

    return train_model, attention_model


# ═══════════════════════════════════════════════════════════════════════════════
#  Bidirectional GRU + Multi-Head Attention (2 heads)
# ═══════════════════════════════════════════════════════════════════════════════

def build_bigru_multihead(vocab_size: int, num_heads: int = 2):
    """
    Same BiGRU backbone but uses MultiHeadAttentionLayer instead.
    attention_weights shape: (batch, num_heads, T).
    """
    inp = tf.keras.layers.Input(shape=(MAX_LEN,), name="token_ids")

    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=EMBED_DIM,
        input_length=MAX_LEN,
        name="embedding",
    )(inp)

    x = tf.keras.layers.SpatialDropout1D(SPATIAL_DROPOUT, name="spatial_drop")(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(GRU_UNITS_1, return_sequences=True, name="gru_1"),
        name="bigru_1",
    )(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(GRU_UNITS_2, return_sequences=True, name="gru_2"),
        name="bigru_2",
    )(x)

    context, attn_weights = MultiHeadAttentionLayer(
        num_heads=num_heads, name="multi_attention"
    )(x)

    logits = _common_head(context, name_prefix="mh_")

    train_model = tf.keras.Model(inputs=inp, outputs=logits,
                                  name="BiGRU_MultiHead")
    attention_model = tf.keras.Model(inputs=inp,
                                      outputs=[logits, attn_weights],
                                      name="BiGRU_MultiHead_viz")

    return train_model, attention_model


# ═══════════════════════════════════════════════════════════════════════════════
#  Utility : print model summary
# ═══════════════════════════════════════════════════════════════════════════════

def print_model_summary(model):
    """Pretty-print the model architecture."""
    print("\n" + "=" * 60)
    print(f"  🧠  Model: {model.name}")
    print("=" * 60)
    model.summary(line_length=90)
