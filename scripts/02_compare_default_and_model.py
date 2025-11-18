#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# scripts/02_compare_default_and_model.py
#
# So sánh model SNGP vs default optimizer trên exec_latencies.csv
# - Tính:
#     + SelAcc: tỷ lệ tham số mà plan do model chọn gần tối ưu (<= 1.1 * optimal)
#     + Speedup_model: trung bình default_latency / model_latency
# - Vẽ biểu đồ bar: Default (=1.0) vs Model (speedup).
#
# Lưu ý:
#   - Dùng cùng kiến trúc + preprocessing như predict_sngp_nearopt.py
#   - Load metadata + plan_cover từ model_dir (metadata.json, plan_cover.json)
#   - KHÔNG dùng model.predict (tránh lỗi optree),
#     mà gọi trực tiếp model(inputs, training=False)
#   - Kiểu dữ liệu input lấy từ metadata['predicates'][i]['data_type']

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

# === Thêm đoạn này để tự chèn project root vào sys.path ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # ~/kepler-sngp
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ==========================================================

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sngp_pipeline.models import ModelConfig, SNGPMultiheadModel
from sngp_pipeline.pipeline_data import (
    DatabaseSimulator,
    DatabaseClient,
    LatencyEstimator,
    PlannedQuery,
)

JSON = Any


# ======================================================================
# Helpers build model (copy logic từ predict_sngp_nearopt, KHÔNG import)
# ======================================================================

def load_metadata_and_cover(model_dir: str) -> Tuple[JSON, List[int]]:
    metadata_path = os.path.join(model_dir, "metadata.json")
    plan_cover_path = os.path.join(model_dir, "plan_cover.json")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(plan_cover_path, "r", encoding="utf-8") as f:
        plan_cover = json.load(f)

    return metadata, plan_cover


def build_model_config_infer(num_plans: int) -> ModelConfig:
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
    preds = metadata.get("predicates", [])
    configs: List[Dict[str, Any]] = []
    for pred in preds:
        dtype = pred.get("data_type")
        if dtype == "float":
            raise NotImplementedError("Float predicates chưa được hỗ trợ ở inference.")
        elif dtype == "int":
            configs.append({"type": "one_hot"})
        else:
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


# ======================================================================
# Group exec_latencies theo bộ tham số
# ======================================================================

def build_param_stats(exec_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    col_params = [c for c in exec_df.columns if c.startswith("param")]
    if not col_params:
        raise ValueError("Không tìm thấy cột param* trong exec_latencies.csv")

    grouped = exec_df.groupby(col_params, sort=False)

    out: Dict[str, Dict[str, Any]] = {}

    print("🔍 Đang group exec_latencies theo bộ tham số ...")
    for params_values, g in grouped:
        if not isinstance(params_values, tuple):
            params_tuple = (params_values,)
        else:
            params_tuple = params_values

        params_str = [str(v) for v in params_tuple]
        param_key = "####".join(params_str)

        g_default = g[g["is_default"] == True]
        if len(g_default) == 0:
            continue
        default_row = g_default.iloc[0]
        default_latency = float(default_row["latency_ms"])
        default_plan = int(default_row["plan_id"])

        idx_best = g["latency_ms"].idxmin()
        best_row = g.loc[idx_best]
        optimal_latency = float(best_row["latency_ms"])
        optimal_plan = int(best_row["plan_id"])

        plan_latencies = {
            int(r["plan_id"]): float(r["latency_ms"]) for _, r in g.iterrows()
        }

        out[param_key] = {
            "params": params_str,
            "default_plan_id": default_plan,
            "default_latency": default_latency,
            "optimal_plan_id": optimal_plan,
            "optimal_latency": optimal_latency,
            "plan_latencies": plan_latencies,
        }

    print(f"✅ Số bộ tham số unique: {len(out)}")
    return out


# ======================================================================
# Predict cho 1 bộ params, dùng metadata['predicates'][i]['data_type']
# ======================================================================

def predict_for_params_single(
    metadata: JSON,
    model: tf.keras.Model,
    plan_cover: List[int],
    params: List[str],
):
    predicates = metadata.get("predicates", [])
    if len(predicates) != len(params):
        raise ValueError(
            f"Số predicates trong metadata ({len(predicates)}) "
            f"không khớp số params ({len(params)})."
        )

    batch_tensors: List[tf.Tensor] = []

    for i, (p_raw, pred_meta, inp) in enumerate(
        zip(params, predicates, model.inputs)
    ):
        tf_dtype = tf.as_dtype(inp.dtype)
        data_type = pred_meta.get("data_type", "text")
        p_str = str(p_raw).strip()

        if tf_dtype.is_integer:
            if data_type == "text":
                distinct_values = pred_meta.get("distinct_values")
                if not distinct_values:
                    raise ValueError(
                        f"Predicate text '{pred_meta.get('name')}' thiếu distinct_values "
                        "trong metadata, không thể map string → id."
                    )
                try:
                    idx = distinct_values.index(p_str)
                except ValueError as e:
                    raise ValueError(
                        f"Giá trị param[{i}]='{p_str}' không nằm trong distinct_values "
                        f"của predicate '{pred_meta.get('name')}'."
                    ) from e
                value = tf.constant([[idx]], dtype=tf_dtype)

            elif data_type in ("int", "integer"):
                try:
                    idx = int(p_str)
                except ValueError as e:
                    raise ValueError(
                        f"Không convert được param[{i}]='{p_str}' sang int "
                        f"cho predicate '{pred_meta.get('name')}'."
                    ) from e
                value = tf.constant([[idx]], dtype=tf_dtype)
            elif data_type == "float":
                try:
                    val = float(p_str)
                except ValueError as e:
                    raise ValueError(
                        f"Không convert được param[{i}]='{p_str}' sang float "
                        f"cho predicate '{pred_meta.get('name')}'."
                    ) from e
                value = tf.constant([[int(val)]], dtype=tf_dtype)
            else:
                try:
                    idx = int(p_str)
                except ValueError as e:
                    raise ValueError(
                        f"Predicate '{pred_meta.get('name')}' (data_type='{data_type}') "
                        f"nhưng input dtype={tf_dtype.name}, param='{p_str}' "
                        "không parse được sang số nguyên."
                    ) from e
                value = tf.constant([[idx]], dtype=tf_dtype)

        elif tf_dtype == tf.string:
            value = tf.constant([[p_str]], dtype=tf.string)

        elif tf_dtype.is_floating:
            try:
                val = float(p_str)
            except ValueError as e:
                raise ValueError(
                    f"Không convert được param[{i}]='{p_str}' sang float "
                    f"cho predicate '{pred_meta.get('name')}'."
                ) from e
            value = tf.constant([[val]], dtype=tf_dtype)
        else:
            value = tf.constant([[p_str]], dtype=tf.string)

        batch_tensors.append(value)

    outputs = model(batch_tensors, training=False)

    if isinstance(outputs, (list, tuple)) and len(outputs) >= 1:
        logits = outputs[0]
    else:
        logits = outputs

    logits_np = np.array(logits)[0]
    probs = 1.0 / (1.0 + np.exp(-logits_np))

    plan_cover_arr = np.array(plan_cover, dtype=np.int32)
    idx_best = int(np.argmax(probs))
    best_plan = int(plan_cover_arr[idx_best])
    best_prob = float(probs[idx_best])

    return best_plan, best_prob, []



# ======================================================================
# Chạy predict_for_params_single cho TẤT CẢ param-set
# ======================================================================

def predict_plan_ids_for_all_params_single(
    metadata: JSON,
    keras_model: tf.keras.Model,
    plan_cover: List[int],
    param_stats: Dict[str, Dict[str, Any]],
) -> Dict[str, int]:
    pred_plan_map: Dict[str, int] = {}
    keys = list(param_stats.keys())
    total = len(keys)

    print(f"[PRED] Tổng số bộ tham số cần predict = {total}")

    num_failed = 0

    for idx, key in enumerate(keys, start=1):
        info = param_stats[key]
        params = info["params"]

        try:
            best_plan, best_prob, _ = predict_for_params_single(
                metadata=metadata,
                model=keras_model,
                plan_cover=plan_cover,
                params=params,
            )
        except Exception as e:
            num_failed += 1
            if num_failed <= 10:
                print(f"[WARN] Bỏ qua param_key={key} vì lỗi khi predict: {e}")
            elif num_failed == 11:
                print("[WARN] ... còn lỗi tương tự, sẽ không in chi tiết nữa.")
            continue

        pred_plan_map[key] = best_plan

        if idx % 50 == 0 or idx == total:
            print(
                f"[PRED] Đã xử lý {idx}/{total} bộ tham số... "
                f"(plan {best_plan}, prob≈{best_prob:.4f})"
            )

    print(
        f"[PRED] Hoàn tất predict cho toàn bộ tham số (single-sample). "
        f"Thành công: {len(pred_plan_map)}, Bị bỏ qua: {num_failed}"
    )
    return pred_plan_map


# ======================================================================
# Đánh giá speedup + SelAcc
# ======================================================================

def evaluate_model_vs_default(
    param_stats: Dict[str, Dict[str, Any]],
    pred_map: Dict[str, int],
    near_optimal_threshold: float = 1.1,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    rows = []

    for key, info in param_stats.items():
        if key not in pred_map:
            continue

        plan_latencies = info["plan_latencies"]
        default_latency = info["default_latency"]
        optimal_latency = info["optimal_latency"]
        optimal_plan_id = info["optimal_plan_id"]

        pred_plan_id = pred_map[key]
        if pred_plan_id not in plan_latencies:
            continue

        model_latency = plan_latencies[pred_plan_id]

        speedup_model = default_latency / model_latency
        speedup_opt = default_latency / optimal_latency
        is_near_opt = model_latency <= near_optimal_threshold * optimal_latency

        rows.append(
            {
                "param_key": key,
                "default_latency": default_latency,
                "optimal_latency": optimal_latency,
                "optimal_plan_id": optimal_plan_id,
                "pred_plan_id": pred_plan_id,
                "model_latency": model_latency,
                "speedup_model": speedup_model,
                "speedup_opt": speedup_opt,
                "selacc_flag": 1 if is_near_opt else 0,
            }
        )

    if not rows:
        raise RuntimeError("Không có dòng nào để đánh giá (rows rỗng).")

    df = pd.DataFrame(rows)
    selacc = df["selacc_flag"].mean()
    speedup_model = df["speedup_model"].mean()
    speedup_opt = df["speedup_opt"].mean()

    summary = {
        "num_points": int(len(df)),
        "selacc": float(selacc),
        "speedup_model": float(speedup_model),
        "speedup_opt": float(speedup_opt),
    }

    return summary, df


# ======================================================================
# Vẽ biểu đồ
# ======================================================================

def plot_speedup(summary: Dict[str, Any], out_plot: str) -> None:
    speedup_model = summary["speedup_model"]
    selacc = summary["selacc"]
    num_points = summary["num_points"]

    labels = ["Default", "Model"]
    values = [1.0, speedup_model]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_ylabel("Speedup vs Default (x)")
    ax.set_title(f"SelAcc={selacc:.3f}, #params={num_points}")

    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_plot), exist_ok=True)
    fig.savefig(out_plot, dpi=120)
    plt.close(fig)
    print(f"📊 Đã lưu biểu đồ speedup vào: {out_plot}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="So sánh SNGP model vs default optimizer trên exec_latencies.csv"
    )
    parser.add_argument(
        "--lat",
        required=True,
        help="Đường dẫn exec_latencies.csv (output từ 01_generate_latencies.py)",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="(KHÔNG còn dùng trực tiếp) Giữ lại cho tương thích CLI.",
    )
    parser.add_argument(
        "--plan-cover",
        required=True,
        help="(KHÔNG còn dùng trực tiếp) Giữ lại cho tương thích CLI.",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Thư mục model SNGP đã train (chứa model_weights.h5, metadata.json, plan_cover.json).",
    )
    parser.add_argument(
        "--out-plot",
        required=True,
        help="Đường dẫn file PNG để vẽ biểu đồ speedup.",
    )
    parser.add_argument(
        "--near-optimal-threshold",
        type=float,
        default=1.1,
        help="Ngưỡng near-optimal (mặc định 1.1).",
    )

    args = parser.parse_args()

    lat_path = args.lat
    model_dir = args.model_dir
    out_plot = args.out_plot
    near_th = args.near_optimal_threshold

    print("=== BẮT ĐẦU SO SÁNH MODEL VS DEFAULT ===")
    print(f"  exec_latencies.csv : {lat_path}")
    print(f"  model_dir          : {model_dir}")
    print(f"  out_plot           : {out_plot}")
    print(f"  near-opt threshold : {near_th}")
    print("  (Lưu ý: --metadata và --plan-cover hiện không dùng, metadata/plan_cover lấy từ model_dir)\n")

    exec_df = pd.read_csv(lat_path)
    print("🔢 exec_df shape:", exec_df.shape)

    metadata, plan_cover = load_metadata_and_cover(model_dir)
    print("📘 query_id:", metadata.get("query_id"))
    print("📘 #predicates:", len(metadata.get("predicates", [])))
    print("📘 #plans (plan_cover):", len(plan_cover))

    preprocessing_config = build_preprocessing_config_for_inference(metadata)
    model_config = build_model_config_infer(num_plans=len(plan_cover))

    print("🧱 Khởi tạo kiến trúc SNGP Multihead để load weights ...")
    sngp_model = SNGPMultiheadModel(
        metadata=metadata,
        plan_ids=plan_cover,
        model_config=model_config,
        preprocessing_config=preprocessing_config,
    )
    keras_model = sngp_model.get_model()

    print("=== Model inputs ===")
    for i, inp in enumerate(keras_model.inputs):
        print(f"  {i}: name={inp.name}, dtype={inp.dtype}, shape={inp.shape}")

    weights_path = os.path.join(model_dir, "model.weights.h5")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Không tìm thấy weights tại: {weights_path}")

    keras_model.load_weights(weights_path)
    print(f"✅ Đã load weights từ: {weights_path}")

    param_stats = build_param_stats(exec_df)

    pred_plan_map = predict_plan_ids_for_all_params_single(
        metadata=metadata,
        keras_model=keras_model,
        plan_cover=plan_cover,
        param_stats=param_stats,
    )

    summary, df_detail = evaluate_model_vs_default(
        param_stats=param_stats,
        pred_map=pred_plan_map,
        near_optimal_threshold=near_th,
    )

    print("\n=== KẾT QUẢ TỔNG HỢP ===")
    print(f"  #params           : {summary['num_points']}")
    print(f"  SelAcc            : {summary['selacc']:.4f}")
    print(f"  Speedup(model)    : {summary['speedup_model']:.4f} x")
    print(f"  Speedup(optimal)  : {summary['speedup_opt']:.4f} x")

    detail_csv = out_plot.replace(".png", "_detail.csv")
    os.makedirs(os.path.dirname(detail_csv), exist_ok=True)
    df_detail.to_csv(detail_csv, index=False)
    print(f"💾 Đã lưu chi tiết so sánh vào: {detail_csv}")

    plot_speedup(summary, out_plot)

    print("=== HOÀN TẤT SO SÁNH MODEL VS DEFAULT ===")


if __name__ == "__main__":
    main()
