"""
=============================================================================
attention.py — Custom Attention Layers for Sequence Classification
=============================================================================
Provides:
  • AttentionLayer          — single-head additive (Bahdanau-style) attention
  • MultiHeadAttentionLayer — manual multi-head variant (2 heads by default)

Both layers accept the full sequence of hidden states from a Bidirectional
RNN and return:
    context_vector   — weighted sum of hidden states  (batch, features)
    attention_weights — softmax weights per timestep   (batch, timesteps)
=============================================================================
"""

import tensorflow as tf


class AttentionLayer(tf.keras.layers.Layer):
    """
    Additive (Bahdanau-style) Attention
    ------------------------------------
    Given a sequence of hidden states  H  of shape (batch, T, D):

        score_t   = tanh(H_t · W + b)  ·  V          (scalar per timestep)
        alpha     = softmax(scores)                    (attention distribution)
        context   = sum_t( alpha_t * H_t )             (weighted combination)

    Trainable parameters:
        W : (D, D)   — projects each hidden state
        b : (D,)     — bias
        V : (D, 1)   — maps projected states to scalars

    Returns
    -------
    context_vector    : (batch, D)
    attention_weights : (batch, T)
    """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape = (batch, timesteps, features)
        feature_dim = int(input_shape[-1])

        self.W = self.add_weight(
            name="attention_W",
            shape=(feature_dim, feature_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_b",
            shape=(feature_dim,),
            initializer="zeros",
            trainable=True,
        )
        self.V = self.add_weight(
            name="attention_V",
            shape=(feature_dim, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : tensor  (batch, timesteps, features)

        Returns
        -------
        context_vector    : (batch, features)
        attention_weights : (batch, timesteps)
        """
        # Score each timestep
        # (batch, T, D) · (D, D) + (D,) → (batch, T, D) → tanh
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=[[-1], [0]]) + self.b)
        # (batch, T, D) · (D, 1) → (batch, T, 1)
        score = tf.tensordot(score, self.V, axes=[[-1], [0]])
        # Squeeze to (batch, T)
        score = tf.squeeze(score, axis=-1)

        # Softmax → attention distribution
        attention_weights = tf.nn.softmax(score, axis=-1)  # (batch, T)

        # Weighted sum of hidden states
        # (batch, T, 1) * (batch, T, D) → sum → (batch, D)
        context_vector = tf.reduce_sum(
            inputs * tf.expand_dims(attention_weights, axis=-1), axis=1
        )

        return context_vector, attention_weights

    def get_config(self):
        return super(AttentionLayer, self).get_config()


class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    """
    Simplified Multi-Head Attention (manual, 2 heads)
    ---------------------------------------------------
    Splits the feature dimension into `num_heads` groups, applies independent
    additive attention to each head, and concatenates the resulting context
    vectors.

    Returns
    -------
    context_vector    : (batch, features)   — concatenation of per-head contexts
    attention_weights : (batch, num_heads, T)
    """

    def __init__(self, num_heads: int = 2, **kwargs):
        super(MultiHeadAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        assert feature_dim % self.num_heads == 0, (
            f"Feature dim ({feature_dim}) must be divisible by num_heads ({self.num_heads})"
        )
        self.head_dim = feature_dim // self.num_heads

        # One set of attention parameters per head
        self.Ws = []
        self.bs = []
        self.Vs = []
        for i in range(self.num_heads):
            self.Ws.append(self.add_weight(
                name=f"head{i}_W", shape=(self.head_dim, self.head_dim),
                initializer="glorot_uniform", trainable=True,
            ))
            self.bs.append(self.add_weight(
                name=f"head{i}_b", shape=(self.head_dim,),
                initializer="zeros", trainable=True,
            ))
            self.Vs.append(self.add_weight(
                name=f"head{i}_V", shape=(self.head_dim, 1),
                initializer="glorot_uniform", trainable=True,
            ))

        super(MultiHeadAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : (batch, T, D)

        Returns
        -------
        context_vector    : (batch, D)
        attention_weights : (batch, num_heads, T)
        """
        batch_size = tf.shape(inputs)[0]
        T = tf.shape(inputs)[1]

        # Split features into heads: (batch, T, num_heads, head_dim)
        heads = tf.reshape(inputs, (batch_size, T, self.num_heads, self.head_dim))

        context_parts = []
        weight_parts = []

        for i in range(self.num_heads):
            h = heads[:, :, i, :]   # (batch, T, head_dim)

            score = tf.nn.tanh(tf.tensordot(h, self.Ws[i], axes=[[-1], [0]]) + self.bs[i])
            score = tf.tensordot(score, self.Vs[i], axes=[[-1], [0]])
            score = tf.squeeze(score, axis=-1)                # (batch, T)
            alpha = tf.nn.softmax(score, axis=-1)             # (batch, T)

            ctx = tf.reduce_sum(h * tf.expand_dims(alpha, -1), axis=1)  # (batch, head_dim)
            context_parts.append(ctx)
            weight_parts.append(alpha)

        context_vector = tf.concat(context_parts, axis=-1)                # (batch, D)
        attention_weights = tf.stack(weight_parts, axis=1)                # (batch, heads, T)

        return context_vector, attention_weights

    def get_config(self):
        config = super(MultiHeadAttentionLayer, self).get_config()
        config.update({"num_heads": self.num_heads})
        return config
