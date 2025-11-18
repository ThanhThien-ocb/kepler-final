# sngp_pipeline/train_sngp_nearopt.py
#
# Train mô hình SNGP Multihead cho bài toán near-optimal classification
# từ exec_latencies.csv + metadata.json + plan_cover.json.
#
# ĐẶC BIỆT:
#   - Tự suy "distinct_values" cho các predicate kiểu text
#     từ exec_latencies.csv, để phù hợp với _apply_preprocessing_layer
#     trong models.py (text → StringLookup + Embedding / OneHot).
#   - Định nghĩa Plans (thay cho KeplerPlanDiscoverer) ngay trong file này,
#     chỉ cần có thuộc tính plan_ids là đủ cho TrainerBase.

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from typing import Any, Dict, List

# ====== RANDOM SEED CHO REPRODUCIBILITY ======
import random
import numpy as np
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

import pandas as pd

from sngp_pipeline.models import ModelConfig, SNGPMultiheadModel
from sngp_pipeline.trainers import NearOptimalClassificationTrainer

JSON = Any


# ======================================================================
# Simple wrapper cho danh sách plan_ids (thay KeplerPlanDiscoverer)
# ======================================================================

@dataclasses.dataclass
class Plans:
    plan_ids: List[int]


# ======================================================================
# Helpers load file
# ======================================================================

def load_metadata(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_plan_cover(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_parameter_column_names(num_params: int) -> List[str]:
    return [f"param{i}" for i in range(num_params)]


# ======================================================================
# Build preprocessing_config + bổ sung distinct_values cho text
# ======================================================================

def enrich_metadata_with_distinct_values(
    metadata: Dict[str, Any],
    exec_df: pd.DataFrame,
) -> None:
    """Bổ sung metadata['predicates'][i]['distinct_values'] cho data_type='text'.

    distinct_values được suy từ cột param_i trong exec_latencies.csv.
    """
    predicates = metadata.get("predicates", [])
    for i, pred in enumerate(predicates):
        dtype = pred.get("data_type")
        col = f"param{i}"
        if dtype == "text":
            if col not in exec_df.columns:
                raise ValueError(f"Thiếu cột {col} trong exec_latencies.csv")

            # Lấy toàn bộ giá trị unique, convert về string
            vals = exec_df[col].astype(str).unique().tolist()
            vals = sorted(set(vals))
            pred["distinct_values"] = vals


def build_preprocessing_config(
    metadata: Dict[str, Any],
    exec_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Sinh preprocessing_config tương ứng với từng predicate.

    Với dataset q1_0, tất cả params là text, ta dùng StringLookup + Embedding.
    Nếu sau này có float/int thì có thể mở rộng thêm.
    """
    predicates = metadata.get("predicates", [])
    configs: List[Dict[str, Any]] = []

    for i, pred in enumerate(predicates):
        dtype = pred.get("data_type")
        col = f"param{i}"

        if dtype == "float":
            # Chuẩn hóa Z-score
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
            # Nếu cần, có thể embed hoặc one_hot. Ở đây chưa dùng cho q1_0.
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
            # Dùng embedding với StringLookup
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


# ======================================================================
# Build ModelConfig
# ======================================================================

def build_model_config(num_plans: int) -> ModelConfig:
    """Tạo ModelConfig cho SNGP multihead.

    num_plans chỉ dùng để tham khảo, không cần trực tiếp trong config.
    """
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


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train SNGP Multihead model cho near-optimal classification."
    )
    parser.add_argument(
        "--lat",
        required=True,
        help="Đường dẫn exec_latencies.csv (đã flatten).",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Đường dẫn metadata.json (query_id + predicates).",
    )
    parser.add_argument(
        "--plan-cover",
        required=True,
        help="Đường dẫn plan_cover.json (list plan_id).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Thư mục output để lưu model/weights.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Số epochs train.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size train.",
    )

    args = parser.parse_args()

    lat_path = args.lat
    metadata_path = args.metadata
    plan_cover_path = args.plan_cover
    out_dir = args.out

    print("=== BẮT ĐẦU TRAIN SNGP NEAR-OPT ===")
    print(f"  exec_latencies.csv : {lat_path}")
    print(f"  metadata.json      : {metadata_path}")
    print(f"  plan_cover.json    : {plan_cover_path}")
    print(f"  out model dir      : {out_dir}")
    print("")

    # 1) Load data
    exec_df = pd.read_csv(lat_path)

    # Đảm bảo thứ tự dòng cố định để train reproducible
    sort_cols = [c for c in ["param0", "param1", "plan_id"] if c in exec_df.columns]
    if sort_cols:
        exec_df = exec_df.sort_values(by=sort_cols).reset_index(drop=True)

    print("🔢 exec_df shape:", exec_df.shape)

    metadata = load_metadata(metadata_path)
    plan_cover = load_plan_cover(plan_cover_path)
    print("query_id:", metadata.get("query_id"))
    print("#predicates:", len(metadata.get("predicates", [])))
    print("#plans (plan_cover):", len(plan_cover))

    # 2) enrich metadata với distinct_values cho text
    enrich_metadata_with_distinct_values(metadata, exec_df)

    # 3) build preprocessing_config
    preprocessing_config = build_preprocessing_config(metadata, exec_df)

    # 4) build ModelConfig
    model_config = build_model_config(num_plans=len(plan_cover))

    # 5) Plans wrapper cho list plan_id
    plans = Plans(plan_ids=plan_cover)

    # 6) Khởi tạo SNGPMultiheadModel
    print("Xây dựng mô hình SNGP Multihead ...")
    model = SNGPMultiheadModel(
        metadata=metadata,
        plan_ids=plans.plan_ids,
        model_config=model_config,
        preprocessing_config=preprocessing_config,
    )

    # 7) Trainer + construct training data
    trainer = NearOptimalClassificationTrainer(
        metadata=metadata,
        plans=plans,
        model=model,
    )

    print("Constructing training data (near-optimal multi-label)...")
# near_optimal_threshold = 1.1 → (1 + τ) với τ = 0.1
    x, y = trainer.construct_training_data(
    exec_df,
    near_optimal_threshold=1.1,
    default_relative=True,
)
    print("[LOG]   #samples:", len(y))
    print("[LOG]   input len:", len(x))


    # 8) Train
    print("Training model ...")
    keras_model = model.get_model()

    # Reset covariance matrix 1 lần trước khi train (cho chắc)
    if hasattr(keras_model, "classifier") and hasattr(
        keras_model.classifier, "reset_covariance_matrix"
    ):
        keras_model.classifier.reset_covariance_matrix()

    history = trainer.train(
        x=x,
        y=y,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # In sơ vài epoch cuối
    hist_dict = history.history
    print("History keys:", list(hist_dict.keys()))
    if "loss" in hist_dict:
        print("Final loss:", hist_dict["loss"][-1])

    # Tìm key chứa 'bin_acc'
    acc_key = None
    for k in hist_dict.keys():
        if "bin_acc" in k:
            acc_key = k
            break
    if acc_key is not None:
        print(f"Final {acc_key}: {hist_dict[acc_key][-1]}")

    # 9) Lưu model weights + metadata/plan_cover
    os.makedirs(out_dir, exist_ok=True)

    weights_path = os.path.join(out_dir, "model.weights.h5")
    keras_model.save_weights(weights_path)
    print(f"Đã lưu model weights vào: {weights_path}")

    # lưu metadata & plan_cover cạnh model cho tiện predict
    meta_out = os.path.join(out_dir, "metadata.json")
    plan_cover_out = os.path.join(out_dir, "plan_cover.json")

    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(plan_cover_out, "w", encoding="utf-8") as f:
        json.dump(plan_cover, f, indent=2)

    print(f"Đã lưu metadata vào: {meta_out}")
    print(f"Đã lưu plan cover vào: {plan_cover_out}")
    print("=== HOÀN TẤT TRAIN SNGP NEAR-OPT ===")


if __name__ == "__main__":
    main()
