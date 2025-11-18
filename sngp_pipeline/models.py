# sngp_pipeline/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras import layers, optimizers


JSON = Any


# ---------------------------------------------------------------------------
# Helper: map data_type string -> tf dtype
# Ở BẢN NÀY: "text" CŨNG ĐƯỢC XỬ LÝ NHƯ "int" (DÙNG ID)
# ---------------------------------------------------------------------------

def get_tf_type(data_type: str) -> tf.DType:
    """Map simple string data types to tf dtypes.

    NOTE: data_type="text" cũng trả về int64 vì ta đã encode text -> id.
    """
    if data_type in ("int", "text"):
        return tf.int64
    if data_type == "float":
        return tf.float32
    raise ValueError(f"Unsupported data_type: {data_type}")


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Stores model architecture / hyperparameters."""

    # Hidden layers
    layer_sizes: List[int]
    dropout_rates: List[float]

    # Optimization
    learning_rate: float
    activation: str

    # Training
    loss: Any  # Loss can be custom function or instance
    metrics: List[Any]

    # SNGP-related
    spectral_norm_multiplier: float = 1.0
    num_gp_random_features: int = 128


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _apply_preprocessing_layer(
    name: str,
    param: Mapping[str, Any],
    input_layer: tf.keras.layers.Layer,
    preprocessing_info: Mapping[str, Any],
) -> tf.keras.layers.Layer:
    """Create preprocessing for a single parameter.

    Supports:
      - float + std_normalization
      - int/text (đã encode thành id) + embedding / one_hot
    """
    data_type = param["data_type"]
    p_type = preprocessing_info["type"]

    # FLOAT + Normalization
    if data_type == "float" and p_type == "std_normalization":
        norm = layers.Normalization(
            mean=preprocessing_info.get("mean", 0.0),
            variance=preprocessing_info.get("variance", 1.0),
            name=f"normalization_{name}",
        )
        return norm(input_layer)

    # INT hoặc TEXT (đã encode thành int id)
    if data_type in ("int", "text"):
        # KHÔNG dùng tf.cast trực tiếp lên KerasTensor nữa
        # vì Input đã có dtype=int64 từ get_tf_type() rồi.
        x = input_layer

        vmin = int(param.get("min", 0))
        vmax = int(param.get("max", vmin))
        vocab_size = vmax - vmin + 1

        shift_const = tf.constant(vmin, dtype=tf.int64)
        shifted = x - shift_const

        if p_type == "embedding":
            emb = layers.Embedding(
                input_dim=vocab_size,
                output_dim=int(preprocessing_info["output_dim"]),
                name=f"embedding_{name}",
            )(shifted)
            return layers.Flatten()(emb)
        elif p_type == "one_hot":
            onehot = layers.CategoryEncoding(
                num_tokens=vocab_size,
                output_mode="one_hot",
                name=f"one_hot_{name}",
            )(shifted)
            return layers.Flatten()(onehot)

    raise ValueError(
        f"Unsupported preprocessing: parameter type={data_type}, preprocessing type={p_type}"
    )


# ---------------------------------------------------------------------------
# Base model class
# ---------------------------------------------------------------------------

class ModelBase:
    """Base class for Kepler-style models."""

    def __init__(
        self,
        metadata: JSON,
        plan_ids: List[int],
        model_config: Optional[ModelConfig],
        preprocessing_config: Sequence[Mapping[str, Any]],
    ):
        if len(metadata["predicates"]) != len(preprocessing_config):
            raise ValueError(
                "Predicates metadata and preprocessing config have mismatched lengths: "
                f"{len(metadata['predicates'])} != {len(preprocessing_config)}"
            )

        # Clone predicates & chuẩn hoá text → có min/max
        predicates = []
        for pred in metadata["predicates"]:
            p = dict(pred)
            if p.get("data_type") == "text":
                # distinct_values phải tồn tại trong metadata.json
                vocab = p.get("distinct_values", [])
                if not vocab:
                    raise ValueError(
                        "Predicate data_type='text' nhưng không có 'distinct_values' trong metadata."
                    )
                p["min"] = 0
                p["max"] = len(vocab) - 1
            predicates.append(p)

        self._predicate_metadata = predicates
        self._num_plans = len(plan_ids)
        self._model_index_to_plan_id = {
            i: plan_id for i, plan_id in enumerate(plan_ids)
        }

        self._model_config = model_config
        self._preprocessing_config = preprocessing_config
        self._inputs: List[tf.keras.Input] = []
        self._model: Optional[tf.keras.Model] = None

    # ---- API ----

    def get_model(self) -> tf.keras.Model:
        if self._model is None:
            raise RuntimeError("Model has not been built yet.")
        return self._model

    # ---- Internal helpers ----

    def _input_layer(
        self,
        data_type: str,
        name: Optional[str] = None,
    ) -> tf.keras.Input:
        """Creates a 1D input layer of the given data type."""
        return Input(shape=(1,), dtype=get_tf_type(data_type), name=name)

    def _construct_preprocessing_layer(self) -> tf.keras.layers.Layer:
        """Creates model inputs + concatenated preprocessing outputs."""
        self._inputs = [
            self._input_layer(p["data_type"], f"param{i}")
            for i, p in enumerate(self._predicate_metadata)
        ]

        to_concat = []
        for i, (p, prep_cfg) in enumerate(
            zip(self._predicate_metadata, self._preprocessing_config)
        ):
            processed = _apply_preprocessing_layer(
                name=f"preprocessing_param{i}",
                param=p,
                input_layer=self._inputs[i],
                preprocessing_info=prep_cfg,
            )
            to_concat.append(processed)

        return layers.Concatenate(name="concat_preprocessed")(to_concat)


# ---------------------------------------------------------------------------
# Simple multi-layer perceptron model (no SNGP)
# ---------------------------------------------------------------------------

class MultiheadModel(ModelBase):
    """Standard MLP model producing one logit/score per plan."""

    def __init__(
        self,
        metadata: JSON,
        plan_ids: List[int],
        model_config: ModelConfig,
        preprocessing_config: Sequence[Mapping[str, Any]],
    ):
        super().__init__(metadata, plan_ids, model_config, preprocessing_config)
        self._build_model()

    def _build_model(self) -> None:
        x = self._construct_preprocessing_layer()

        for i, (size, dr) in enumerate(
            zip(self._model_config.layer_sizes, self._model_config.dropout_rates)
        ):
            x = layers.Dense(
                size,
                activation=self._model_config.activation,
                name=f"dense_{i}",
            )(x)
            x = layers.Dropout(dr, name=f"dropout_{i}")(x)

        outputs = layers.Dense(
            self._num_plans,
            name="output_dense",
        )(x)

        model = Model(inputs=self._inputs, outputs=outputs, name="multihead_mlp")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self._model_config.learning_rate),
            loss=self._model_config.loss,
            metrics=self._model_config.metrics,
        )
        self._model = model


# ---------------------------------------------------------------------------
# Spectral Normalization
# ---------------------------------------------------------------------------

class SpectralNormalization(layers.Wrapper):
    """Spectral normalization wrapper for Dense / Embedding layers."""

    def __init__(
        self,
        layer: layers.Layer,
        power_iterations: int = 1,
        norm_multiplier: float = 1.0,
        **kwargs,
    ):
        if power_iterations <= 0:
            raise ValueError(
                f"`power_iterations` must be > 0, got {power_iterations}"
            )
        super().__init__(layer, **kwargs)
        self.power_iterations = power_iterations
        self.norm_multiplier = norm_multiplier

    def build(self, input_shape):
        super().build(input_shape)

        # Choose kernel attribute
        if hasattr(self.layer, "kernel"):
            self.kernel = self.layer.kernel
        elif hasattr(self.layer, "embeddings"):
            self.kernel = self.layer.embeddings
        else:
            raise ValueError(
                f"Layer {type(self.layer).__name__} has no `kernel` or `embeddings`."
            )

        self.kernel_shape = self.kernel.shape

        self.u = self.add_weight(
            shape=(1, self.kernel_shape[-1]),
            initializer="random_normal",
            trainable=False,
            name="sn_u",
            dtype=self.kernel.dtype,
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        if training:
            u, w = self._compute_spectral_normalized_weights()
            self.u.assign(u)
            self.kernel.assign(w)

        return self.layer(inputs)

    def _compute_spectral_normalized_weights(
        self,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        w = tf.reshape(self.kernel, (-1, self.kernel_shape[-1]))
        u = self.u

        for _ in range(self.power_iterations):
            v = tf.linalg.l2_normalize(tf.matmul(u, w, transpose_b=True))
            u = tf.linalg.l2_normalize(tf.matmul(v, w))

        sigma = tf.matmul(tf.matmul(v, w), u, transpose_b=True)
        w_sn = (self.kernel / sigma) * self.norm_multiplier
        w_sn = tf.reshape(w_sn, self.kernel_shape)

        return tf.cast(u, self.u.dtype), tf.cast(w_sn, self.kernel.dtype)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "power_iterations": self.power_iterations,
                "norm_multiplier": self.norm_multiplier,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Laplace approximation for GP covariance
# ---------------------------------------------------------------------------

class LaplaceRandomFeatureCovariance(layers.Layer):
    """Maintains precision matrix for random feature GP via Laplace approximation."""

    def __init__(
        self,
        momentum: float = 0.999,
        ridge_penalty: float = 1.0,
        likelihood: str = "gaussian",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.momentum = momentum
        self.ridge_penalty = ridge_penalty
        self.likelihood = likelihood

    def build(self, input_shape):
        dim = int(input_shape[-1])
        eye = tf.eye(dim, dtype=self.dtype) * self.ridge_penalty
        self.precision_matrix = self.add_weight(
            name="gp_precision_matrix",
            shape=(dim, dim),
            initializer=tf.keras.initializers.Constant(eye),
            trainable=False,
        )
        self.initial_precision_matrix = eye
        super().build(input_shape)

    def call(self, gp_feature, logits=None, training=False):
        """Update/compute covariance via Laplace approximation."""
        batch_size = tf.cast(tf.shape(gp_feature)[0], gp_feature.dtype)

        # Prob multiplier dạng (B, 1)
        if logits is None:
            prob_multiplier = tf.ones(
                (tf.shape(gp_feature)[0], 1), dtype=gp_feature.dtype
            )
        elif self.likelihood == "binary_logistic":
            prob = tf.sigmoid(logits)                  # (B, C)
            var = prob * (1.0 - prob)                  # (B, C)
            prob_multiplier = tf.reduce_mean(var, axis=-1, keepdims=True)  # (B,1)
        elif self.likelihood == "poisson":
            rate = tf.exp(logits)                      # (B, C)
            prob_multiplier = tf.reduce_mean(rate, axis=-1, keepdims=True)
        else:  # gaussian
            prob_multiplier = tf.ones(
                (tf.shape(gp_feature)[0], 1), dtype=gp_feature.dtype
            )

        # (B, D) * (B,1) -> (B, D)
        gp_feature_adjusted = gp_feature * tf.sqrt(
            tf.cast(prob_multiplier, gp_feature.dtype)
        )

        precision_matrix_minibatch = tf.matmul(
            gp_feature_adjusted, gp_feature_adjusted, transpose_a=True
        )

        if self.momentum > 0:
            precision_matrix_minibatch /= batch_size
            precision_matrix_new = (
                self.momentum * self.precision_matrix
                + (1.0 - self.momentum) * precision_matrix_minibatch
            )
        else:
            precision_matrix_new = self.precision_matrix + precision_matrix_minibatch

        self.precision_matrix.assign(precision_matrix_new)

        if training:
            # During training chỉ cần update stats, trả về eye placeholder
            return tf.eye(tf.shape(gp_feature)[0], dtype=self.dtype)

        cov = tf.linalg.inv(self.precision_matrix)
        return tf.matmul(gp_feature, tf.matmul(cov, gp_feature, transpose_b=True))

    def reset_precision_matrix(self):
        """Reset covariance stats (called at start of epochs)."""
        self.precision_matrix.assign(self.initial_precision_matrix)


# ---------------------------------------------------------------------------
# Random Feature Gaussian Process layer
# ---------------------------------------------------------------------------

class RandomFeatureGaussianProcess(layers.Layer):
    """Random feature GP layer for SNGP models."""

    def __init__(
        self,
        units: int,
        num_inducing: int = 1024,
        gp_kernel_scale: float = 1.0,
        gp_output_bias: float = 0.0,
        normalize_input: bool = False,
        scale_random_features: bool = True,
        l2_regularization: float = 1e-6,
        cov_momentum: float = 0.999,
        cov_ridge_penalty: float = 1.0,
        likelihood: str = "gaussian",
        return_covariance: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.num_inducing = num_inducing
        self.gp_input_scale = 1.0 / np.sqrt(float(gp_kernel_scale))
        self.normalize_input = normalize_input
        self.scale_random_features = scale_random_features
        self.gp_feature_scale = np.sqrt(2.0 / float(num_inducing))
        self.l2_regularization = l2_regularization
        self.gp_output_bias_value = gp_output_bias
        self.cov_momentum = cov_momentum
        self.cov_ridge_penalty = cov_ridge_penalty
        self.likelihood = likelihood
        self.return_covariance = return_covariance

    def build(self, input_shape):
        # Optional input norm
        if self.normalize_input:
            self.input_norm = layers.LayerNormalization()
            self.input_norm.build(input_shape)
            feature_input_shape = self.input_norm.compute_output_shape(input_shape)
        else:
            self.input_norm = None
            feature_input_shape = input_shape

        # Random feature map: Dense + cos
        self.random_dense = layers.Dense(
            self.num_inducing,
            use_bias=True,
            kernel_initializer="random_normal",
            name="gp_random_dense",
        )
        self.random_dense.build(feature_input_shape)
        rf_shape = self.random_dense.compute_output_shape(feature_input_shape)

        # Covariance layer
        if self.return_covariance:
            self.cov_layer = LaplaceRandomFeatureCovariance(
                momentum=self.cov_momentum,
                ridge_penalty=self.cov_ridge_penalty,
                likelihood=self.likelihood,
                dtype=self.dtype,
                name="gp_covariance",
            )
            self.cov_layer.build(rf_shape)
        else:
            self.cov_layer = None

        # Linear output head on top of random features
        self.output_layer = layers.Dense(
            self.units,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization),
            name="gp_output_dense",
        )
        self.output_layer.build(rf_shape)

        self.output_bias = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(self.gp_output_bias_value),
            trainable=True,
            name="gp_output_bias",
        )

        super().build(input_shape)

    def call(self, inputs, training: bool = False):
        x = inputs
        if self.normalize_input and self.input_norm is not None:
            x = self.input_norm(x)
        else:
            x = x * tf.cast(self.gp_input_scale, x.dtype)

        # Random features
        rf = self.random_dense(x)
        rf = tf.math.cos(rf)
        if self.scale_random_features:
            rf *= tf.cast(self.gp_feature_scale, rf.dtype)

        gp_output = self.output_layer(rf) + self.output_bias

        if not self.return_covariance or self.cov_layer is None:
            return gp_output, None

        cov = self.cov_layer(rf, logits=gp_output, training=training)
        return gp_output, cov

    def reset_covariance_matrix(self):
        if self.cov_layer is not None:
            self.cov_layer.reset_precision_matrix()


# ---------------------------------------------------------------------------
# SNGP Model wrapper + callback
# ---------------------------------------------------------------------------

class ResetCovarianceCallback(tf.keras.callbacks.Callback):
    """Reset GP covariance matrix at the beginning of each epoch (except 0)."""

    def on_epoch_begin(self, epoch, logs=None):
        if epoch > 0 and hasattr(self.model, "classifier"):
            classifier = getattr(self.model, "classifier")
            if hasattr(classifier, "reset_covariance_matrix"):
                classifier.reset_covariance_matrix()


class SNGPModel(tf.keras.Model):
    """Wrapper Keras Model that always uses ResetCovarianceCallback in fit()."""

    def __init__(self, classifier: RandomFeatureGaussianProcess, **kwargs):
        super().__init__(**kwargs)
        self.classifier = classifier

    def fit(self, *args, **kwargs):
        cb = kwargs.get("callbacks", [])
        cb = list(cb) + [ResetCovarianceCallback()]
        kwargs["callbacks"] = cb
        return super().fit(*args, **kwargs)


# ---------------------------------------------------------------------------
# SNGP Multihead model
# ---------------------------------------------------------------------------

class SNGPMultiheadModel(ModelBase):
    """Spectral-normalized Neural Gaussian Process model."""

    def __init__(
        self,
        metadata: JSON,
        plan_ids: List[int],
        model_config: ModelConfig,
        preprocessing_config: Sequence[Mapping[str, Any]],
    ):
        super().__init__(metadata, plan_ids, model_config, preprocessing_config)
        self._build_model()

    def _build_model(self) -> None:
        x = self._construct_preprocessing_layer()

        # Hidden layers with Spectral Normalization
        for i, (size, dr) in enumerate(
            zip(self._model_config.layer_sizes, self._model_config.dropout_rates)
        ):
            dense = layers.Dense(
                size,
                activation=self._model_config.activation,
                name=f"intermediate_dense_{i}",
            )
            sn = SpectralNormalization(
                dense,
                power_iterations=1,
                norm_multiplier=self._model_config.spectral_norm_multiplier,
                name=f"spectral_norm_{i}",
            )
            x = sn(x)
            x = layers.Dropout(dr, name=f"dropout_{i}")(x)

        # GP output layer
        gp_layer = RandomFeatureGaussianProcess(
            units=self._num_plans,
            num_inducing=self._model_config.num_gp_random_features,
            normalize_input=False,
            scale_random_features=True,
            cov_momentum=-1.0,           # exact covariance accumulation
            cov_ridge_penalty=1.0,
            likelihood="binary_logistic",
            name="output_gp_layer",
        )

        logits, covariance = gp_layer(x, training=False)

        model = SNGPModel(
            classifier=gp_layer,
            inputs=self._inputs,
            outputs=[logits, covariance],
            name="sngp_multihead_model",
        )

        # Chỉ áp loss/metrics lên logits; covariance không có loss.
        model.compile(
            optimizer=optimizers.Adam(
                learning_rate=self._model_config.learning_rate
            ),
            loss=[self._model_config.loss, None],
            metrics=[[m for m in self._model_config.metrics], []],
        )

        self._model = model
