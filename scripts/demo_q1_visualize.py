#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
demo_q1_visualize.py

Visualize đơn giản cho q1_0:
- Đọc exec_latencies.csv + metadata.json + plan_cover.json + model.weights.h5
- Chọn một vài bộ param
- So sánh latency:
    Default plan vs Model predicted plan vs Best plan
- Vẽ bar chart và lưu PNG cho mỗi ví dụ.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import random
from typing import Any, Dict, List, Tuple
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # thư mục /home/kepler/kepler-sngp

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sngp_pipeline.models import ModelConfig, SNGPMultiheadModel

JSON = Any

# ---------------------------------------------------------
# Helper: load file
# ---------------------------------------------------------
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------
# Các hàm giống file train_sngp_nearopt.py
#  - enrich_metadata_with_distinct_values
#  - build_preprocessing_config
#  - build_model_config
# ---------------------------------------------------------
def enrich_metadata_with_distinct_values(
    metadata: Dict[str, Any],
    exec_df: pd.DataFrame,
) -> None:
    """Bổ sung metadata['predicates'][i]['distinct_values'] cho data_type='text'."""
    predicates = metadata.get("predicates", [])
    for i, pred in enumerate(predicates):
        dtype = pred.get("data_type")
        col = f"param{i}"
        if dtype == "text":
            if col not in exec_df.columns:
                raise ValueError(f"Thiếu cột {col} trong exec_latencies.csv")

            vals = exec_df[col].astype(str).unique().tolist()
            vals = sorted(set(vals))
            pred["distinct_values"] = vals


def build_preprocessing_config(
    metadata: Dict[str, Any],
    exec_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Sinh preprocessing_config tương ứng với từng predicate."""
    predicates = metadata.get("predicates", [])
    configs: List[Dict[str, Any]] = []

    for i, pred in enumerate(predicates):
        dtype = pred.get("data_type")
        col = f"param{i}"

        if dtype == "float":
            if col not in exec_df.columns:
                raise ValueError(f"Thiếu cột {col} trong exec_latencies.csv")
            series = exec_df[col].astype(float)
            mean = float(series.mean())
            var = float(series.var())
            if var <= 0:
                var = 1.0
            configs.append(
                {
                    "type": "std_normalization",
                    "mean": mean,
                    "variance": var,
                }
            )
        elif dtype == "int":
            if col not in exec_df.columns:
                raise ValueError(f"Thiếu cột {col} trong exec_latencies.csv")
            series = exec_df[col].astype(int)
            min_v = int(series.min())
            max_v = int(series.max())
            pred["min"] = min_v
            pred["max"] = max_v
            configs.append(
                {
                    "type": "one_hot",
                }
            )
        else:
            # text
            if "distinct_values" not in pred:
                raise ValueError(
                    f"Predicate text {pred.get('name')} chưa có distinct_values"
                )
            configs.append(
                {
                    "type": "embedding",
                    "output_dim": 16,
                    "num_oov_indices": 1,
                }
            )

    if len(configs) != len(predicates):
        raise ValueError(
            "Độ dài preprocessing_config không khớp với số predicate "
            f"({len(configs)} vs {len(predicates)})"
        )

    return configs


def build_model_config(num_plans: int) -> ModelConfig:
    """Config mô hình SNGP Multihead – giống file train_sngp_nearopt.py."""
    layer_sizes = [64, 64]
    dropout_rates = [0.1, 0.1]

    return ModelConfig(
        layer_sizes=layer_sizes,
        dropout_rates=dropout_rates,
        learning_rate=1e-3,
        activation="relu",
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="bin_acc")],
        spectral_norm_multiplier=0.9,
        num_gp_random_features=128,
    )


# ---------------------------------------------------------
# Encode 1 bộ param -> input cho model
# ---------------------------------------------------------
def build_vocab_maps(metadata: Dict[str, Any]) -> Dict[int, Dict[str, int]]:
    """Tạo vocab map cho các predicate text từ metadata.distinct_values."""
    vocab_maps: Dict[int, Dict[str, int]] = {}
    for idx, pred in enumerate(metadata.get("predicates", [])):
        if pred.get("data_type") == "text":
            vocab = pred.get("distinct_values", [])
            if not vocab:
                raise ValueError(
                    f"Predicate index {idx} data_type='text' nhưng không có distinct_values."
                )
            vocab_maps[idx] = {str(v): i for i, v in enumerate(vocab)}
    return vocab_maps


def encode_params_for_model(
    metadata: Dict[str, Any],
    vocab_maps: Dict[int, Dict[str, int]],
    param_values: List[Any],
) -> List[np.ndarray]:
    """param_values = [param0, param1, ...] -> list input numpy arrays (shape (1,1))."""
    inputs: List[np.ndarray] = []

    for i, pred in enumerate(metadata.get("predicates", [])):
        dtype = pred.get("data_type")
        v = param_values[i]

        if dtype == "int":
            arr = np.array([[int(v)]], dtype=np.int64)
            inputs.append(arr)
        elif dtype == "float":
            arr = np.array([[float(v)]], dtype=np.float32)
            inputs.append(arr)
        else:
            # text -> id
            vocab_map = vocab_maps.get(i, {})
            s = "" if v is None else str(v)
            idx_id = vocab_map.get(s, 0)
            arr = np.array([[idx_id]], dtype=np.int64)
            inputs.append(arr)

    return inputs


# ---------------------------------------------------------
# Chọn các param_key để demo
# ---------------------------------------------------------
def pick_param_keys_with_full_plans(
    df: pd.DataFrame,
    plan_cover: List[int],
    max_examples: int = 3,
) -> List[str]:
    """Chọn một vài param_key mà có đủ tất cả plan_id trong plan_cover."""
    col_params = [c for c in df.columns if c.startswith("param")]
    if not col_params:
        raise ValueError("Không tìm thấy cột param* trong exec_latencies.csv")

    df = df.copy()
    df["param_key"] = df[col_params].astype(str).agg("####".join, axis=1)

    groups = df.groupby("param_key")
    candidates: List[str] = []

    for pk, g in groups:
        have_plans = set(int(p) for p in g["plan_id"].unique())
        if set(plan_cover).issubset(have_plans):
            candidates.append(pk)

    random.shuffle(candidates)
    return candidates[:max_examples]


# ---------------------------------------------------------
# Tính default / best / model plan cho 1 param_key
# ---------------------------------------------------------
def get_default_and_best(
    group: pd.DataFrame,
) -> Tuple[int, float, int, float]:
    """Từ một group (1 param_key), lấy:
    - default_plan_id, default_latency
    - best_plan_id (min latency), best_latency
    """
    # default = dòng is_default == True
    g_default = group[group["is_default"] == True]
    if g_default.empty:
        # fallback: lấy plan_id có latency min làm default (trường hợp đặc biệt)
        idx = group["latency_ms"].idxmin()
        row_def = group.loc[idx]
    else:
        row_def = g_default.iloc[0]

    default_plan_id = int(row_def["plan_id"])
    default_latency = float(row_def["latency_ms"])

    # best = latency nhỏ nhất
    idx_best = group["latency_ms"].idxmin()
    row_best = group.loc[idx_best]
    best_plan_id = int(row_best["plan_id"])
    best_latency = float(row_best["latency_ms"])

    return default_plan_id, default_latency, best_plan_id, best_latency


def get_latency_of_plan(
    group: pd.DataFrame,
    plan_id: int,
) -> float:
    """Lấy latency_ms của 1 plan_id trong group."""
    sub = group[group["plan_id"] == plan_id]
    if sub.empty:
        return float("nan")
    return float(sub.iloc[0]["latency_ms"])


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Demo visualize Default vs Model vs Best cho q1_0."
    )
    parser.add_argument("--lat", required=True, help="exec_latencies.csv dùng để train.")
    parser.add_argument("--metadata", required=True, help="metadata.json.")
    parser.add_argument("--plan-cover", required=True, help="plan_cover.json.")
    parser.add_argument("--weights", required=True, help="model.weights.h5.")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Thư mục output để lưu hình PNG.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=3,
        help="Số ví dụ muốn vẽ.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=== DEMO VISUALIZE Q1_0 ===")
    print(f"  lat        : {args.lat}")
    print(f"  metadata   : {args.metadata}")
    print(f"  plan_cover : {args.plan_cover}")
    print(f"  weights    : {args.weights}")
    print(f"  out_dir    : {args.out_dir}")
    print("")

    # 1) Load data
    exec_df = pd.read_csv(args.lat)
    # sort lại để ổn định
    sort_cols = [c for c in ["param0", "param1", "plan_id"] if c in exec_df.columns]
    if sort_cols:
        exec_df = exec_df.sort_values(by=sort_cols).reset_index(drop=True)

    print("exec_df shape:", exec_df.shape)

    metadata = load_json(args.metadata)
    plan_cover: List[int] = load_json(args.plan_cover)
    print("query_id      :", metadata.get("query_id"))
    print("#predicates   :", len(metadata.get("predicates", [])))
    print("#plans (cover):", len(plan_cover))

    # 2) enrich metadata + preprocessing_config
    enrich_metadata_with_distinct_values(metadata, exec_df)
    preprocessing_config = build_preprocessing_config(metadata, exec_df)
    model_config = build_model_config(num_plans=len(plan_cover))

    # 3) Build model + load weights
    print("Xây dựng mô hình SNGP Multihead ...")
    model = SNGPMultiheadModel(
        metadata=metadata,
        plan_ids=plan_cover,
        model_config=model_config,
        preprocessing_config=preprocessing_config,
    )
    keras_model = model.get_model()
    keras_model.load_weights(args.weights)
    print("Đã load weights xong.")

    # 4) Vocab map để encode text
    vocab_maps = build_vocab_maps(metadata)

    # 5) Chọn 1 vài param_key để demo
    param_keys = pick_param_keys_with_full_plans(
        exec_df, plan_cover, max_examples=args.num_examples
    )
    if not param_keys:
        raise RuntimeError("Không tìm được param_key nào có đủ plan_cover.")

    print(f"Sẽ visualize {len(param_keys)} ví dụ.")

    # chuẩn bị group theo param_key
    col_params = [c for c in exec_df.columns if c.startswith("param")]
    exec_df = exec_df.copy()
    exec_df["param_key"] = exec_df[col_params].astype(str).agg("####".join, axis=1)
    groups = exec_df.groupby("param_key")

    for idx, pk in enumerate(param_keys, start=1):
        g = groups.get_group(pk).copy()

        # lấy default / best
        d_pid, d_lat, b_pid, b_lat = get_default_and_best(g)

        # lấy param values (dùng 1 row bất kỳ)
        row0 = g.iloc[0]
        param_values = [row0[c] for c in col_params]

        # encode param -> input cho model
        x_inputs = encode_params_for_model(metadata, vocab_maps, param_values)

        # predict
        logits, _ = keras_model.predict(x_inputs, verbose=0)
        logits = logits[0]
        best_idx = int(np.argmax(logits))
        pred_plan_id = int(plan_cover[best_idx])
        pred_lat = get_latency_of_plan(g, pred_plan_id)

        print(f"\n=== Ví dụ #{idx} ===")
        print("param_key:", pk)
        for j, c in enumerate(col_params):
            print(f"  {c} =", param_values[j])
        print(f"  Default plan: {d_pid}, latency = {d_lat:.3f} ms")
        print(f"  Model plan  : {pred_plan_id}, latency = {pred_lat:.3f} ms")
        print(f"  Best plan   : {b_pid}, latency = {b_lat:.3f} ms")

        # 6) Vẽ bar chart
        labels = ["Default", "Model", "Best"]
        values = [d_lat, pred_lat, b_lat]

        plt.figure(figsize=(4, 4))
        plt.bar(labels, values)
        plt.ylabel("Latency (ms)")
        plt.title(f"Q1_0 – Example {idx}")

        # thêm text trên đầu cột
        for i, v in enumerate(values):
            plt.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

        out_path = os.path.join(args.out_dir, f"q1_example_{idx}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"  → Đã lưu hình: {out_path}")

    print("\n=== HOÀN TẤT DEMO VISUALIZE ===")


if __name__ == "__main__":
    main()
