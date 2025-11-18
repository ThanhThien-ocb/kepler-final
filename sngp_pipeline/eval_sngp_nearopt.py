# sngp_pipeline/eval_sngp_nearopt.py
#
# Đánh giá model SNGP trên exec_latencies.csv:
#  - SelAcc (τ = 1.1)
#  - Speedup vs default
#  - Regression rate
#
# Dùng lại cùng kiến trúc SNGP như lúc train, rồi load_weights.
#
# Cách chạy:
#   python -m sngp_pipeline.eval_sngp_nearopt \
#       --model-dir models/sngp_nearopt_q1_0 \
#       --lat data/exec_latencies.csv \
#       --tau 1.1

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from sngp_pipeline.models import (
    ModelConfig,
    SNGPMultiheadModel,
)

JSON = Any


# ---------------------------------------------------------------------
# Helpers load metadata / plan_cover / build model
# ---------------------------------------------------------------------

def load_metadata_and_cover(model_dir: str) -> Tuple[JSON, List[int]]:
    metadata_path = os.path.join(model_dir, "metadata.json")
    plan_cover_path = os.path.join(model_dir, "plan_cover.json")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(plan_cover_path, "r", encoding="utf-8") as f:
        plan_cover = json.load(f)

    return metadata, plan_cover


def build_model_config(num_plans: int) -> ModelConfig:
    layer_sizes = [64, 64]
    dropout_rates = [0.1, 0.1]
    return ModelConfig(
        layer_sizes=layer_sizes,
        dropout_rates=dropout_rates,
        learning_rate=1e-3,
        activation="relu",
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="output_gp_layer_bin_acc")],
        spectral_norm_multiplier=0.9,
        num_gp_random_features=128,
    )


def build_preprocessing_config_for_inference(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Dùng metadata (đã có distinct_values) để tạo preprocessing_config.

    Với q1_0: tất cả predicates là text → StringLookup + Embedding.
    """
    preds = metadata.get("predicates", [])
    configs: List[Dict[str, Any]] = []

    for pred in preds:
        dtype = pred.get("data_type")
        if dtype == "float":
            # Chưa support float ở inference (chưa có mean/var lưu), nếu cần có thể bổ sung sau
            raise NotImplementedError("Float predicates chưa được hỗ trợ ở inference.")
        elif dtype == "int":
            # Đơn giản: one_hot (nếu sau này cần)
            configs.append({"type": "one_hot"})
        else:
            # text
            if "distinct_values" not in pred:
                raise ValueError("Predicate text thiếu distinct_values trong metadata.")
            configs.append(
                {
                    "type": "embedding",
                    "output_dim": 16,
                    "num_oov_indices": 1,
                }
            )

    return configs


def build_and_load_sngp_model(model_dir: str) -> Tuple[tf.keras.Model, List[int]]:
    metadata, plan_cover = load_metadata_and_cover(model_dir)
    preprocessing_config = build_preprocessing_config_for_inference(metadata)
    model_config = build_model_config(num_plans=len(plan_cover))

    sngp = SNGPMultiheadModel(
        metadata=metadata,
        plan_ids=plan_cover,
        model_config=model_config,
        preprocessing_config=preprocessing_config,
    )
    keras_model = sngp.get_model()

    weights_path = os.path.join(model_dir, "model.weights.h5")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Không tìm thấy {weights_path}, hãy chạy train trước.")

    keras_model.load_weights(weights_path)
    return keras_model, plan_cover


# ---------------------------------------------------------------------
# Xây dataset theo từng bộ tham số
# ---------------------------------------------------------------------

def build_param_groups(exec_df: pd.DataFrame):
    """Group exec_latencies theo bộ tham số param*."""
    param_cols = [c for c in exec_df.columns if c.startswith("param")]
    if not param_cols:
        raise ValueError("Không tìm thấy cột param* trong exec_latencies.csv")

    grouped = exec_df.groupby(param_cols, sort=False)

    keys: List[Tuple[str, ...]] = []
    lat_opt_list: List[float] = []
    lat_def_list: List[float] = []
    plan_opt_list: List[int] = []
    lat_tables: List[Dict[int, float]] = []

    for params, group in grouped:
        if not isinstance(params, tuple):
            params = (params,)

        keys.append(tuple(params))

        # optimal
        idx_opt = group["latency_ms"].idxmin()
        best_row = group.loc[idx_opt]
        lat_opt = float(best_row["latency_ms"])
        plan_opt = int(best_row["plan_id"])

        lat_opt_list.append(lat_opt)
        plan_opt_list.append(plan_opt)

        # default
        default_rows = group[group["is_default"] == True]
        if default_rows.empty:
            default_row = group.iloc[0]
        else:
            default_row = default_rows.iloc[0]
        lat_def = float(default_row["latency_ms"])
        lat_def_list.append(lat_def)

        # map plan_id -> latency
        lat_map: Dict[int, float] = {}
        for _, r in group.iterrows():
            pid = int(r["plan_id"])
            lat_map[pid] = float(r["latency_ms"])
        lat_tables.append(lat_map)

    return {
        "param_cols": param_cols,
        "keys": keys,
        "lat_opt": np.array(lat_opt_list, dtype=np.float64),
        "lat_def": np.array(lat_def_list, dtype=np.float64),
        "plan_opt": np.array(plan_opt_list, dtype=np.int32),
        "lat_tables": lat_tables,
    }


def build_model_inputs_from_keys(
    keys: List[Tuple[str, ...]],
    num_params: int,
):
    """Tạo list numpy arrays (mỗi param là shape [N, 1]) từ list param keys."""
    N = len(keys)
    inputs: List[np.ndarray] = []
    for i in range(num_params):
        vals = [k[i] for k in keys]
        arr = np.array(vals, dtype=str).reshape(N, 1)
        inputs.append(arr)
    return inputs


# ---------------------------------------------------------------------
# Evaluate metrics
# ---------------------------------------------------------------------

def evaluate_model(
    model: tf.keras.Model,
    plan_cover: List[int],
    exec_df: pd.DataFrame,
    tau: float = 1.1,
    batch_size: int = 512,
):
    info = build_param_groups(exec_df)
    param_cols = info["param_cols"]
    keys = info["keys"]
    lat_opt = info["lat_opt"]
    lat_def = info["lat_def"]
    plan_opt = info["plan_opt"]
    lat_tables = info["lat_tables"]

    num_params = len(param_cols)
    num_queries = len(keys)
    print(f"#queries (distinct param combinations): {num_queries}")
    print(f"#params: {num_params}")
    print(f"param columns: {param_cols}")
    print(f" #plans in cover: {len(plan_cover)}")

    # Tạo model inputs
    model_inputs = build_model_inputs_from_keys(keys, num_params)

    # Predict
    print("🔮 Đang cho model predict toàn bộ workload ...")
    pred = model.predict(model_inputs, batch_size=batch_size, verbose=1)
    if isinstance(pred, (list, tuple)) and len(pred) >= 1:
        logits = pred[0]
    else:
        logits = pred

    probs = 1.0 / (1.0 + np.exp(-logits))
    plan_cover_arr = np.array(plan_cover, dtype=np.int32)
    chosen_idx = np.argmax(probs, axis=1)
    chosen_plan_ids = plan_cover_arr[chosen_idx]

    lat_model = np.zeros(num_queries, dtype=np.float64)
    nearopt_flags = np.zeros(num_queries, dtype=bool)
    regression_flags = np.zeros(num_queries, dtype=bool)
    speedup_vals = np.zeros(num_queries, dtype=np.float64)

    for i in range(num_queries):
        lat_map = lat_tables[i]
        pid = int(chosen_plan_ids[i])

        # nếu vì lý do nào đó plan không có trong lat_map → fallback default
        if pid in lat_map:
            lm = lat_map[pid]
        else:
            lm = lat_def[i]
        lat_model[i] = lm

        # near-opt: lm <= tau * lat_opt
        nearopt_flags[i] = (lm <= tau * lat_opt[i])

        # regression: slo hơn default
        regression_flags[i] = (lm > lat_def[i])

        # speedup vs default
        if lm > 0:
            speedup_vals[i] = lat_def[i] / lm
        else:
            speedup_vals[i] = 1.0

    # Metrics
    selacc = float(np.mean(nearopt_flags))
    mean_speedup = float(np.mean(speedup_vals))
    median_speedup = float(np.median(speedup_vals))
    regression_rate = float(np.mean(regression_flags))

    # baseline default vs optimal
    default_nearopt_flags = lat_def <= tau * lat_opt
    default_selacc = float(np.mean(default_nearopt_flags))
    default_speedup_to_opt = float(np.mean(lat_def / lat_opt))

    print("\n===== KẾT QUẢ ĐÁNH GIÁ SNGP (NEAR-OPT) =====")
    print(f"τ (near-opt threshold): {tau}")
    print(f"SelAcc(model)       : {selacc:.4f}")
    print(f"SelAcc(default)     : {default_selacc:.4f}  (default so với optimal)")
    print(f"Mean speedup vs def.: {mean_speedup:.4f}")
    print(f"Median speedup vs def.: {median_speedup:.4f}")
    print(f"Regression rate     : {regression_rate:.4f}")
    print(f"Mean default/optimal speedup: {default_speedup_to_opt:.4f}")
    print("=============================================")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Đánh giá model SNGP near-opt trên exec_latencies.csv."
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Thư mục model (chứa model_weights.h5, metadata.json, plan_cover.json).",
    )
    parser.add_argument(
        "--lat",
        required=True,
        help="Đường dẫn exec_latencies.csv.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.1,
        help="Ngưỡng near-opt τ (mặc định 1.1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size predict.",
    )

    args = parser.parse_args()

    print("=== BẮT ĐẦU EVAL SNGP NEAR-OPT ===")
    print(f"  model dir : {args.model_dir}")
    print(f"  lat file  : {args.lat}")
    print(f"  tau       : {args.tau}")
    print("")

    exec_df = pd.read_csv(args.lat)
    model, plan_cover = build_and_load_sngp_model(args.model_dir)

    evaluate_model(
        model=model,
        plan_cover=plan_cover,
        exec_df=exec_df,
        tau=args.tau,
        batch_size=args.batch_size,
    )

    print("=== HOÀN TẤT EVAL ===")


if __name__ == "__main__":
    main()
