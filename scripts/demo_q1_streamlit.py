#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
demo_q1_streamlit.py

Web UI đơn giản để demo mô hình SNGP cho q1_0:

- Cho phép chọn param0 (site) và param1 (tag) từ exec_latencies.csv
- Model dự đoán plan tốt → so sánh với:
    + plan default
    + plan tối ưu thực tế (min latency)

Chạy:
  streamlit run scripts/demo_q1_streamlit.py
"""

from __future__ import annotations

from typing import Any, Dict, List
import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

CURRENT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sngp_pipeline.models import ModelConfig, SNGPMultiheadModel

JSON = Any

# ================== CONFIG ĐƯỜNG DẪN ==================
LAT_PATH = "data/exec_latencies.csv"          # file bạn đã convert
MODEL_DIR = "models/sngp_nearopt_q1_0"        # thư mục khi train

METADATA_PATH   = os.path.join(MODEL_DIR, "metadata.json")
PLAN_COVER_PATH = os.path.join(MODEL_DIR, "plan_cover.json")
WEIGHTS_PATH    = os.path.join(MODEL_DIR, "model.weights.h5")


# ================== HÀM PHỤ ==================
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
            # text: đã có distinct_values trong metadata (do lúc train đã enrich)
            if "distinct_values" not in pred:
                raise ValueError(
                    f"Predicate text {pred.get('name')} "
                    "chưa có distinct_values trong metadata.json"
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
    """Giống file train_sngp_nearopt.py."""
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


def build_vocab_maps(metadata: Dict[str, Any]) -> Dict[int, Dict[str, int]]:
    """Tạo map: index predicate -> {value_string -> id} (giống trainer)."""
    vocab_maps: Dict[int, Dict[str, int]] = {}
    for idx, pred in enumerate(metadata.get("predicates", [])):
        if pred.get("data_type") == "text":
            vocab = pred.get("distinct_values", [])
            vocab_maps[idx] = {str(v): i for i, v in enumerate(vocab)}
    return vocab_maps


def encode_params_to_inputs(
    params: List[Any],
    metadata: Dict[str, Any],
    vocab_maps: Dict[int, Dict[str, int]],
) -> List[np.ndarray]:
    """Encode list param values -> list input tensor (giống construct_training_data)."""
    X: List[np.ndarray] = []
    predicates = metadata.get("predicates", [])

    for i, val in enumerate(params):
        pred = predicates[i]
        dtype = pred.get("data_type")

        if dtype == "int":
            arr = np.asarray([[int(val)]], dtype=np.int64)
            X.append(arr)

        elif dtype == "float":
            arr = np.asarray([[float(val)]], dtype=np.float32)
            X.append(arr)

        elif dtype == "text":
            vm = vocab_maps.get(i, {})
            s = "" if val is None else str(val)
            idx_id = vm.get(s, 0)  # nếu không có trong vocab, map về 0
            arr = np.asarray([[idx_id]], dtype=np.int64)
            X.append(arr)
        else:
            raise ValueError(f"Unsupported data_type for features: {dtype}")

    return X


# ================== CACHE TẢI MODEL & DATA ==================
@st.cache_resource
def load_all():
    # 1) Exec latencies
    exec_df = pd.read_csv(LAT_PATH)
    sort_cols = [c for c in ["param0", "param1", "plan_id"] if c in exec_df.columns]
    if sort_cols:
        exec_df = exec_df.sort_values(by=sort_cols).reset_index(drop=True)

    # 2) metadata + plan_cover
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(PLAN_COVER_PATH, "r", encoding="utf-8") as f:
        plan_cover = json.load(f)

    # 3) preprocessing_config + model_config
    preprocessing_config = build_preprocessing_config(metadata, exec_df)
    model_config = build_model_config(num_plans=len(plan_cover))

    # 4) xây dựng model SNGP + load weights
    model = SNGPMultiheadModel(
        metadata=metadata,
        plan_ids=plan_cover,
        model_config=model_config,
        preprocessing_config=preprocessing_config,
    )
    keras_model = model.get_model()

    # Load weights mềm (phòng khi lệch chút)
    keras_model.load_weights(
        WEIGHTS_PATH,
        by_name=True,
        skip_mismatch=True,
    )

    # 5) vocab maps
    vocab_maps = build_vocab_maps(metadata)

    return exec_df, metadata, plan_cover, keras_model, vocab_maps


# ================== STREAMLIT APP ==================
def main():
    st.set_page_config(
        page_title="Kepler q1_0 SNGP Demo",
        layout="wide",
    )

    st.title("🔍 Kepler SNGP Demo – Query q1_0 (StackOverflow Tags)")
    st.write(
        "Demo nhỏ: chọn **(site, tag)** → model đề xuất **plan** gần tối ưu, "
        "so sánh với plan **default** và **tối ưu thực tế** (min latency)."
    )

    # Load mọi thứ (đã cache)
    exec_df, metadata, plan_cover, keras_model, vocab_maps = load_all()

    # ====== Lấy danh sách param0 / param1 từ exec_latencies.csv ======
    all_sites = sorted(exec_df["param0"].unique().tolist())
    default_site = "stackoverflow" if "stackoverflow" in all_sites else all_sites[0]

    col_left, col_right = st.columns(2)

    # 1. Chọn tham số (param)
    with col_left:
        st.header("1️⃣ Chọn tham số (param) ↔")

        # Chọn site trước
        site = st.selectbox(
            "param0 (site)",
            all_sites,
            index=all_sites.index(default_site) if default_site in all_sites else 0,
        )

        # Sau khi chọn site → chỉ lấy tag có data cho site đó
        tags_for_site = sorted(
            exec_df.loc[exec_df["param0"] == site, "param1"].unique().tolist()
        )
        tag = st.selectbox("param1 (tag)", tags_for_site)

        # Nút predict
        predict_btn = st.button("🔍 Dự đoán plan & so sánh")

    with col_right:
        st.subheader("ℹ️ Thông tin model & dữ liệu")
        st.markdown(
            f"""
            - **Số dòng exec_latencies**: `{len(exec_df):,}`
            - **Số plan trong plan_cover**: `{len(plan_cover)}`
            - **Query ID**: `{metadata.get("query_id", "q1_0")}`
            """
        )

    st.markdown("---")

    if not predict_btn:
        return

    # ================== PREDICT ==================
    # Lọc ra tất cả dòng tương ứng (site, tag) trong exec_latencies
    mask = (exec_df["param0"] == site) & (exec_df["param1"] == tag)
    df_key = exec_df[mask].copy()

    if df_key.empty:
        st.error("Không tìm thấy dữ liệu thực nghiệm cho cặp param đã chọn.")
        return

    # Tính plan tối ưu thực tế (min latency)
    best_idx = df_key["latency_ms"].idxmin()
    best_row = df_key.loc[best_idx]
    best_plan_id = int(best_row["plan_id"])
    best_latency = float(best_row["latency_ms"])

    # Lấy plan default (is_default == True)
    df_default = df_key[df_key["is_default"] == True]
    if df_default.empty:
        default_plan_id = None
        default_latency = None
    else:
        def_row = df_default.iloc[0]
        default_plan_id = int(def_row["plan_id"])
        default_latency = float(def_row["latency_ms"])

    # Encode input cho model
    params = [site, tag]
    X_list = encode_params_to_inputs(params, metadata, vocab_maps)

    # Predict (SNGP model trả về [logits, covariance])
    logits, covariance = keras_model.predict(X_list, verbose=0)
    scores = logits[0]  # (num_plans,)

    # Plan model chọn (theo score cao nhất)
    model_idx = int(np.argmax(scores))
    model_plan_id = int(plan_cover[model_idx])

    df_model = df_key[df_key["plan_id"] == model_plan_id]
    if df_model.empty:
        model_latency = None
    else:
        model_latency = float(df_model["latency_ms"].iloc[0])

    # ============== HIỂN THỊ TÓM TẮT ==============
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "✅ Plan tối ưu (thực nghiệm)",
        f"plan {best_plan_id}",
        f"{best_latency:.3f} ms",
    )

    if default_plan_id is not None and default_latency is not None:
        c2.metric(
            "⚙️ Plan default (optimizer)",
            f"plan {default_plan_id}",
            f"{default_latency:.3f} ms",
        )
    else:
        c2.write("⚙️ Không có plan default trong dữ liệu.")

    if model_latency is not None:
        # So sánh với default & optimal
        if default_latency is not None and model_latency > 0:
            speedup_vs_default = default_latency / model_latency
            delta_vs_default = f"{speedup_vs_default:.2f}× nhanh hơn default"
        else:
            delta_vs_default = "N/A"

        if best_latency > 0:
            slow_vs_best = model_latency / best_latency
            delta_vs_best = f"{slow_vs_best:.2f}× chậm hơn tối ưu"
        else:
            delta_vs_best = "N/A"

        c3.metric(
            "🧠 Plan model đề xuất",
            f"plan {model_plan_id}",
            f"{model_latency:.3f} ms",
        )

        st.success(
            f"**Model chọn plan `{model_plan_id}`** – "
            f"latency ≈ **{model_latency:.3f} ms**.\n\n"
            f"- So với **default**: {delta_vs_default}\n"
            f"- So với **tối ưu**: {delta_vs_best}"
        )
    else:
        c3.write("🧠 Model chọn plan không có trong exec_latencies (không đo).")

    st.markdown("---")


if __name__ == "__main__":
    main()
