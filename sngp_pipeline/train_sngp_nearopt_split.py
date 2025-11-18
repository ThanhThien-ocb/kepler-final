#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train SNGP Multihead model (NEAR-OPTIMAL CLASSIFICATION)
với cách chia dữ liệu train/test 80% – 20%.

Dùng:
  python sngp_pipeline/train_sngp_nearopt_split.py \
      --lat data/exec_latencies.csv \
      --metadata data/metadata.json \
      --plan-cover data/plan_cover.json \
      --out models/sngp_nearopt_q1_0_split \
      --epochs 30 \
      --batch-size 128 \
      --test-split 0.2
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from typing import Any, Dict, List

import random
import numpy as np
import tensorflow as tf
import pandas as pd

from sngp_pipeline.models import ModelConfig, SNGPMultiheadModel
from sngp_pipeline.trainers import NearOptimalClassificationTrainer

# ==== RANDOM SEED ====
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

JSON = Any


# ======================================================================
# Simple wrapper cho danh sách plan_ids (thay KeplerPlanDiscoverer)
# ======================================================================
@dataclasses.dataclass
class Plans:
    plan_ids: List[int]


# ======================================================================
# Load helper
# ======================================================================
def load_metadata(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_plan_cover(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ======================================================================
# Build preprocessing_config + distinct_values cho text
# ======================================================================
def enrich_metadata_with_distinct_values(
    metadata: Dict[str, Any],
    exec_df: pd.DataFrame,
) -> None:
    """Bổ sung giá trị distinct cho param dạng text."""
    predicates = metadata.get("predicates", [])
    for i, pred in enumerate(predicates):
        if pred.get("data_type") == "text":
            col = f"param{i}"
            vals = sorted(set(exec_df[col].astype(str).unique().tolist()))
            pred["distinct_values"] = vals


def build_preprocessing_config(
    metadata: Dict[str, Any],
    exec_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Sinh config cho từng param."""
    predicates = metadata.get("predicates", [])
    configs = []

    for i, pred in enumerate(predicates):
        dtype = pred["data_type"]
        col = f"param{i}"

        if dtype == "float":
            series = exec_df[col].astype(float)
            mean = float(series.mean())
            var = float(series.var()) if series.var() > 0 else 1.0
            configs.append({
                "type": "std_normalization",
                "mean": mean,
                "variance": var,
            })

        elif dtype == "int":
            series = exec_df[col].astype(int)
            pred["min"] = int(series.min())
            pred["max"] = int(series.max())
            configs.append({"type": "one_hot"})

        else:  # text
            if "distinct_values" not in pred:
                raise ValueError("metadata text thiếu distinct_values")
            configs.append({
                "type": "embedding",
                "output_dim": 16,
                "num_oov_indices": 1,
            })

    return configs


# ======================================================================
# Build model config
# ======================================================================
def build_model_config(num_plans: int) -> ModelConfig:
    return ModelConfig(
        layer_sizes=[64, 64],
        dropout_rates=[0.1, 0.1],
        learning_rate=1e-3,
        activation="relu",
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="bin_acc")],
        spectral_norm_multiplier=0.9,
        num_gp_random_features=128,
    )


# ======================================================================
# MAIN
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train SNGP Multihead model với train/test split."
    )
    parser.add_argument("--lat", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--plan-cover", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Tỷ lệ test (vd: 0.2 nghĩa là 20% dữ liệu dùng để test)",
    )

    args = parser.parse_args()

    print("=== TRAIN SNGP – Train/Test Split ===")
    print(f"LAT        : {args.lat}")
    print(f"METADATA   : {args.metadata}")
    print(f"PLAN COVER : {args.plan_cover}")
    print(f"OUTPUT DIR : {args.out}")
    print(f"TEST SPLIT : {args.test_split}")
    print()

    # 1) Load exec latencies
    exec_df = pd.read_csv(args.lat)
    exec_df = exec_df.sort_values(
        by=[c for c in ["param0", "param1", "plan_id"] if c in exec_df.columns]
    ).reset_index(drop=True)

    # 2) Load metadata & plan cover
    metadata = load_metadata(args.metadata)
    plan_cover = load_plan_cover(args.plan_cover)

    # 3) Enrich metadata with distinct_values
    enrich_metadata_with_distinct_values(metadata, exec_df)

    # 4) Build preprocessing & model config
    preprocessing_config = build_preprocessing_config(metadata, exec_df)
    model_config = build_model_config(len(plan_cover))

    # 5) Model + trainer
    plans = Plans(plan_cover)

    model = SNGPMultiheadModel(
        metadata=metadata,
        plan_ids=plan_cover,
        model_config=model_config,
        preprocessing_config=preprocessing_config,
    )
    trainer = NearOptimalClassificationTrainer(metadata, plans, model)

    # 6) Construct data
    print("Constructing training data...")
    x, y = trainer.construct_training_data(exec_df)
    print(f"Total samples: {len(y)}")

    # ===== SPLIT 80/20 =====
    test_split = args.test_split
    idx = np.arange(len(y))
    np.random.shuffle(idx)

    split_point = int(len(y) * (1 - test_split))
    train_idx, test_idx = idx[:split_point], idx[split_point:]

    x_train = [feat[train_idx] for feat in x]
    x_test  = [feat[test_idx]  for feat in x]
    y_train = y[train_idx]
    y_test  = y[test_idx]

    print(f"Train samples: {len(y_train)}")
    print(f"Test samples : {len(y_test)}")

    # 7) Train
    keras_model = model.get_model()
    if hasattr(keras_model, "classifier"):
        if hasattr(keras_model.classifier, "reset_covariance_matrix"):
            keras_model.classifier.reset_covariance_matrix()

    print("\n=== TRAINING ===")
    history = trainer.train(
        x=x_train,
        y=y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # 8) Evaluate test set
    print("\n=== TEST EVALUATION ===")
    results = keras_model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=0)
    print("Test results:", results)

    # 9) Save model + metadata
    os.makedirs(args.out, exist_ok=True)

    weights_path = os.path.join(args.out, "model.weights.h5")
    keras_model.save_weights(weights_path)
    print(f"Saved weights → {weights_path}")

    # Save metadata
    with open(os.path.join(args.out, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save plan cover
    with open(os.path.join(args.out, "plan_cover.json"), "w", encoding="utf-8") as f:
        json.dump(plan_cover, f, indent=2)

    print("=== DONE ===")


if __name__ == "__main__":
    main()
