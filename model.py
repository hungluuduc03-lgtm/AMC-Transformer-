"""
vit_modulation_core.py
----------------------
Minimal Vision Transformer (ViT) model and preprocessing utilities
for radio modulation classification.

Included:
  - Serializable custom Keras layers: `Patches`, `PatchEncoder`
  - `ViTConfig` dataclass for hyperparameters
  - `create_vit_classifier(cfg)` to build & compile the model
  - Preprocessing: training-channel stats + normalization layer
  - Save/Load helpers with custom objects registered

Excluded:
  - Plotting, calibration metrics, evaluation utilities

Python 3.9+, TensorFlow 2.10+ recommended.
License: MIT
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable


# ------------------------------
#  Serializable custom layers
# ------------------------------

@register_keras_serializable(package="custom")
class Patches(layers.Layer):
    """Extract non-overlapping patches along the time axis.

    Input shape: [B, 2, T, 1]  (I/Q channels, time, singleton spatial)
    Uses a (2 x patch_size) window with strides (2, patch_size), so each patch
    spans both I and Q and a contiguous time segment.
    """

    def __init__(self, patch_size: int, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = int(patch_size)

    def call(self, images: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, 2, self.patch_size, 1],
            strides=[1, 2, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = tf.shape(patches)[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self) -> Dict:
        base = super().get_config()
        base.update({"patch_size": self.patch_size})
        return base


@register_keras_serializable(package="custom")
class PatchEncoder(layers.Layer):
    """Linear projection + learnable positional embeddings for patches."""

    def __init__(self, num_patches: int, projection_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = int(num_patches)
        self.projection_dim = int(projection_dim)
        self.projection = layers.Dense(units=self.projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches, output_dim=self.projection_dim
        )

    def call(self, patch: tf.Tensor) -> tf.Tensor:
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patch) + self.position_embedding(positions)

    def get_config(self) -> Dict:
        base = super().get_config()
        base.update(
            {
                "num_patches": self.num_patches,
                "projection_dim": self.projection_dim,
            }
        )
        return base


# ------------------------------
#  Model builder
# ------------------------------

def _mlp(x: tf.Tensor, hidden_units: Sequence[int], dropout_rate: float) -> tf.Tensor:
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


@dataclass
class ViTConfig:
    # Input tensor shape for Keras Input
    input_shape: Tuple[int, int, int] = (2, 1024, 1)  # [channels, time, spatial=1]
    # Patch/encoder
    patch_size: int = 16
    projection_dim: int = 64
    num_heads: int = 4
    transformer_layers: int = 6
    transformer_units: Tuple[int, int] = (128, 64)  # MLP width inside transformer blocks
    # Classifier head
    mlp_head_units: Tuple[int, ...] = (128,)
    num_classes: int = 24
    # Optimization
    learning_rate: float = 1e-3
    # Regularization
    attn_dropout: float = 0.1
    mlp_dropout: float = 0.1
    head_dropout: float = 0.5

    def num_patches(self) -> int:
        # Given input [2, T, 1], with sizes (2, patch_size) and strides (2, patch_size)
        # number of patches along time = T // patch_size
        T = self.input_shape[1]
        return T // self.patch_size


def create_vit_classifier(cfg: ViTConfig) -> keras.Model:
    """Create and compile a Vision Transformer classifier."""
    inputs = layers.Input(shape=cfg.input_shape)
    patches = Patches(cfg.patch_size, name="patches")(inputs)
    encoded = PatchEncoder(cfg.num_patches(), cfg.projection_dim, name="patch_encoder")(patches)

    x = encoded
    for i in range(cfg.transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{i}")(x)
        attn = layers.MultiHeadAttention(
            num_heads=cfg.num_heads, key_dim=cfg.projection_dim, dropout=cfg.attn_dropout, name=f"mha_{i}"
        )(x1, x1)
        x2 = layers.Add(name=f"skip_attn_{i}")([attn, x])
        x3 = layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{i}")(x2)
        x3 = _mlp(x3, hidden_units=cfg.transformer_units, dropout_rate=cfg.mlp_dropout)
        x = layers.Add(name=f"skip_mlp_{i}")([x3, x2])

    rep = layers.LayerNormalization(epsilon=1e-6, name="ln_final")(x)
    rep = layers.Flatten(name="flatten")(rep)
    rep = layers.Dropout(cfg.head_dropout, name="head_dropout")(rep)
    features = _mlp(rep, hidden_units=cfg.mlp_head_units, dropout_rate=cfg.head_dropout)
    logits = layers.Dense(cfg.num_classes, name="logits")(features)
    outputs = layers.Activation("softmax", name="softmax")(logits)

    model = keras.Model(inputs=inputs, outputs=outputs, name="vit_modulation")
    optimizer = keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


# ------------------------------
#  Preprocessing
# ------------------------------

def compute_train_channel_stats(x_train_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute channel-wise mean/std after per-sample normalization.
    Args:
        x_train_np: Array of shape [N, 2, T, 1].
    Returns:
        (mu[2], std[2]) as float32.
    """
    x = np.squeeze(x_train_np, axis=-1)  # [N, 2, T]
    x = x - x.mean(axis=2, keepdims=True)  # DC removal per sample
    rms = np.sqrt((x ** 2).sum(axis=(1, 2)) / (x.shape[1] * x.shape[2]) + 1e-8)
    x = x / (rms[:, None, None] + 1e-8)
    mu = x.mean(axis=(0, 2)).astype("float32")   # [2]
    std = x.std(axis=(0, 2)).astype("float32")   # [2]
    return mu, std


def make_normalize_layer(mu: np.ndarray, std: np.ndarray):
    """Build a tf.function normalization op using training-set stats."""
    mu_tf = tf.constant(mu[None, :, None, None], dtype=tf.float32)   # [1, 2, 1, 1]
    std_tf = tf.constant(std[None, :, None, None], dtype=tf.float32)
    eps = tf.constant(1e-8, dtype=tf.float32)

    @tf.function
    def _norm(x: tf.Tensor) -> tf.Tensor:
        x = x - tf.reduce_mean(x, axis=2, keepdims=True)
        rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=[1, 2, 3], keepdims=True) + eps)
        x = x / (rms + eps)
        x = (x - mu_tf) / (std_tf + eps)
        return x

    return _norm


# ------------------------------
#  Save/Load with custom objects
# ------------------------------

def save_model(model: keras.Model, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model.save(path)


def load_model(path: str) -> keras.Model:
    return keras.models.load_model(path, custom_objects={"Patches": Patches, "PatchEncoder": PatchEncoder})


__all__ = [
    "Patches",
    "PatchEncoder",
    "ViTConfig",
    "create_vit_classifier",
    "compute_train_channel_stats",
    "make_normalize_layer",
    "save_model",
    "load_model",
]
