# sngp_pipeline/predict_sngp_nearopt.py
#
# Interface test: dự đoán plan cho một bộ tham số cụ thể.
#
# Cách chạy (q1_0 có 2 param: site, tag):
#   python -m sngp_pipeline.predict_sngp_nearopt \
#       --model-dir models/sngp_nearopt_q1_0 \
#       --param stackoverflow \
#       --param iron-router \
#       --lat data/exec_latencies.csv
#
# Nếu không truyền --lat thì chỉ in plan + probabilities.

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
# Helpers build model (giống eval)
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


def build_and_load_sngp_model(model_dir: str) -> Tuple[tf.keras.Model, List[int], JSON]:
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
    return keras_model, plan_cover, metadata


# ---------------------------------------------------------------------
# Predict + phân tích latency (nếu có exec_latencies)
# ---------------------------------------------------------------------
def predict_for_params(
    metadata: JSON,
    model: tf.keras.Model,
    plan_cover: List[int],
    params: List[str],
):
    """
    Dự đoán cho một bộ params (list strings) dựa trên:
      - metadata['predicates'][i]['data_type']
      - metadata['predicates'][i]['distinct_values'] (cho text)
      - model.inputs[i].dtype (thực tế: int64)

    Nếu input dtype = int64 và predicate là text:
      → map string → index trong distinct_values → feed index (int64)
    """
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

        # Chuẩn hóa param về string để lookup trong distinct_values
        p_str = str(p_raw).strip()

        # Case 1: model input là integer (int64 / int32)
        if tf_dtype.is_integer:
            if data_type == "text":
                # Dùng distinct_values để map string -> index
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
                # Numeric thực sự
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
                # Nếu input vẫn int64 nhưng data_type=float, cast về int (hiếm)
                value = tf.constant([[int(val)]], dtype=tf_dtype)
            else:
                # Fallback: cố parse số, nếu không được thì lỗi
                try:
                    idx = int(p_str)
                except ValueError as e:
                    raise ValueError(
                        f"Predicate '{pred_meta.get('name')}' (data_type='{data_type}') "
                        f"nhưng input dtype={tf_dtype.name}, param='{p_str}' "
                        "không parse được sang số nguyên."
                    ) from e
                value = tf.constant([[idx]], dtype=tf_dtype)

        # Case 2: model input là string
        elif tf_dtype == tf.string:
            value = tf.constant([[p_str]], dtype=tf.string)

        # Case 3: model input là float
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
            # Dtype lạ, fallback sang string
            value = tf.constant([[p_str]], dtype=tf.string)

        batch_tensors.append(value)

    # Gọi model trực tiếp (bypass model.predict + optree)
    outputs = model(batch_tensors, training=False)

    # Model SNGP: outputs = [logits, covariance] hoặc chỉ logits
    if isinstance(outputs, (list, tuple)) and len(outputs) >= 1:
        logits = outputs[0]
    else:
        logits = outputs

    logits_np = np.array(logits)[0]  # (num_plans,)
    probs = 1.0 / (1.0 + np.exp(-logits_np))  # sigmoid

    plan_cover_arr = np.array(plan_cover, dtype=np.int32)
    idx_best = int(np.argmax(probs))
    best_plan = int(plan_cover_arr[idx_best])
    best_prob = float(probs[idx_best])

    # Top-k (ở đây k = min(5, num_plans))
    k = min(5, len(plan_cover_arr))
    sorted_idx = np.argsort(probs)[::-1][:k]

    topk = []
    for i in sorted_idx:
        pid = int(plan_cover_arr[i])
        pr = float(probs[i])
        topk.append((pid, pr))

    return best_plan, best_prob, topk


def analyze_latency_for_params(
    exec_df: pd.DataFrame,
    params: List[str],
):
    """Tìm default/optimal/model latency cho bộ params (nếu có)."""
    # Giả sử exec_df có cột param0, param1, ...
    param_cols = [c for c in exec_df.columns if c.startswith("param")]
    if len(param_cols) != len(params):
        raise ValueError(
            f"Số param truyền vào ({len(params)}) không khớp số cột param* ({len(param_cols)})"
        )

    # Lọc các dòng có bộ params đúng
    cond = None
    for i, col in enumerate(param_cols):
        c = exec_df[col].astype(str) == str(params[i])
        cond = c if cond is None else (cond & c)

    sub = exec_df[cond]
    if sub.empty:
        print("⚠ Không tìm thấy dòng nào trong exec_latencies.csv với bộ params này.")
        return None

    # Optimal: latency nhỏ nhất
    idx_opt = sub["latency_ms"].idxmin()
    row_opt = sub.loc[idx_opt]

    # Default:
    if "is_default" in sub.columns:
        rows_def = sub[sub["is_default"] == True]
        if rows_def.empty:
            row_def = sub.iloc[0]
        else:
            row_def = rows_def.iloc[0]
    else:
        # Không có cột is_default: lấy dòng đầu tiên
        row_def = sub.iloc[0]

    info = {
        "param_cols": param_cols,
        "sub_df": sub,
        "optimal": {
            "plan_id": int(row_opt["plan_id"]),
            "latency_ms": float(row_opt["latency_ms"]),
        },
        "default": {
            "plan_id": int(row_def["plan_id"]),
            "latency_ms": float(row_def["latency_ms"]),
        },
    }
    return info




# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interface test: predict SNGP near-opt cho một bộ params."
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Thư mục model (chứa model_weights.h5, metadata.json, plan_cover.json).",
    )
    parser.add_argument(
        "--param",
        action="append",
        required=True,
        help="Tham số query (có thể truyền nhiều lần, theo thứ tự param0, param1, ...)",
    )
    parser.add_argument(
        "--lat",
        help="(Tuỳ chọn) exec_latencies.csv để xem thêm default/optimal latency.",
    )

    args = parser.parse_args()

    params = args.param
    print("=== PREDICT SNGP NEAR-OPT ===")
    print(f"  model dir : {args.model_dir}")
    print(f"  params    : {params}")
    if args.lat:
        print(f"  lat file  : {args.lat}")
    print("")

    model, plan_cover, metadata = build_and_load_sngp_model(args.model_dir)

    print("=== Model inputs ===")
    for i, inp in enumerate(model.inputs):
        print(f"  {i}: name={inp.name}, dtype={inp.dtype}, shape={inp.shape}")
    print("")

    best_plan, best_prob, topk = predict_for_params(metadata, model, plan_cover, params)

    print(" KẾT QUẢ DỰ ĐOÁN:")
    print(f"  Plan được chọn: {best_plan} (prob ≈ {best_prob:.4f})")
    print("  Top-k plan (plan_id, prob):")
    for pid, pr in topk:
        print(f"     - plan {pid}: {pr:.4f}")

    if args.lat:
        exec_df = pd.read_csv(args.lat)
        info = analyze_latency_for_params(exec_df, params)
        if info is not None:
            opt = info["optimal"]
            dft = info["default"]

            print("\n Thông tin latency trong exec_latencies.csv:")
            print(
                f"  - Default plan  : id={dft['plan_id']}, "
                f"latency={dft['latency_ms']:.4f} ms"
            )
            print(
                f"  - Optimal plan  : id={opt['plan_id']}, "
                f"latency={opt['latency_ms']:.4f} ms"
            )

            # latency plan model chọn
            sub = info["sub_df"]
            row_model = sub[sub["plan_id"] == best_plan]
            if not row_model.empty:
                lm = float(row_model.iloc[0]["latency_ms"])
                print(
                    f"  - Model plan    : id={best_plan}, "
                    f"latency={lm:.4f} ms"
                )
                print(
                    f"    * speedup vs default = {dft['latency_ms'] / lm:.4f}x"
                )
                print(
                    f"    * near-opt (τ=1.1)? "
                    f"{'YES' if lm <= 1.1 * opt['latency_ms'] else 'NO'}"
                )
            else:
                print(
                    "  - Không tìm thấy latency của plan model chọn trong exec_latencies.csv."
                )

    print("=== HOÀN TẤT PREDICT ===")


if __name__ == "__main__":
    main()
