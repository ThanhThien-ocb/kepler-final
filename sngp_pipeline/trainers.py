# sngp_pipeline/trainers.py
from __future__ import annotations

from typing import Any, List, Tuple, Mapping, Dict
import numpy as np
import pandas as pd

JSON = Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_parameter_column_names(parameter_count: int) -> List[str]:
    """param0, param1, ..."""
    return [f"param{i}" for i in range(parameter_count)]


def cast_df_columns(
    df: pd.DataFrame,
    predicate_metadata: List[Mapping[str, Any]],
) -> pd.DataFrame:
    """Ép kiểu các cột param theo metadata['predicates'].

    Ở đây chỉ ép kiểu "int"/"float" (text sẽ được encode riêng trong trainer).
    """
    df = df.copy()
    for i, pred in enumerate(predicate_metadata):
        col = f"param{i}"
        if col not in df.columns:
            continue
        dtype = pred.get("data_type")
        if dtype == "int":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif dtype == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        # text: giữ nguyên string, encode sau
    return df


# ---------------------------------------------------------------------------
# Base Trainer
# ---------------------------------------------------------------------------

class TrainerBase:
    """Base trainer class cho pipeline SNGP."""

    def __init__(self, metadata: JSON, plans: Any, model: Any):
        """
        Args:
          metadata: metadata.json (có field 'predicates')
          plans: có thể là
            - list/tuple các plan_id
            - hoặc object có thuộc tính .plan_ids (KeplerPlanDiscoverer)
          model: instance của ModelBase (MultiheadModel / SNGPMultiheadModel)
        """
        self._predicate_metadata = metadata["predicates"]

        # Chuẩn hoá list plan_id
        if isinstance(plans, (list, tuple)):
            plan_ids = list(plans)
        elif hasattr(plans, "plan_ids"):
            plan_ids = list(plans.plan_ids)
        else:
            raise TypeError(
                "TrainerBase expects `plans` to be a list of plan_ids "
                "hoặc object có thuộc tính `.plan_ids`."
            )

        self._plan_ids: List[int] = plan_ids
        self._model = model

        # map plan_id -> index trong output head
        self._plan_id_to_index = {pid: i for i, pid in enumerate(self._plan_ids)}

        # Build vocab map cho các predicate kiểu text:
        # self._text_vocab_maps[i] = { value: idx }
        self._text_vocab_maps: Dict[int, Dict[str, int]] = {}
        for idx, pred in enumerate(self._predicate_metadata):
            if pred.get("data_type") == "text":
                vocab = pred.get("distinct_values", [])
                if not vocab:
                    raise ValueError(
                        f"Predicate index {idx} data_type='text' nhưng không có distinct_values."
                    )
                self._text_vocab_maps[idx] = {str(v): i for i, v in enumerate(vocab)}

    def apply_preprocessing(self, df: pd.DataFrame) -> None:
        """Hook nếu sau này cần xử lý đặc biệt (timestamp...)."""
        return

    def get_parameter_column_names(self) -> List[str]:
        return get_parameter_column_names(len(self._predicate_metadata))

    def train(
        self,
        x: List[Any],
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        sample_weight: np.ndarray | None = None,
    ):
        """Gọi keras_model.fit() thông qua ModelBase."""
        keras_model = self._model.get_model()
        return keras_model.fit(
            x=x,
            y=y,
            epochs=epochs,
            batch_size=batch_size,
            sample_weight=sample_weight,
        )

    def construct_training_data(
        self, query_execution_df: pd.DataFrame
    ) -> Tuple[List[Any], np.ndarray]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Near-Optimal Multi-label Classification Trainer
# ---------------------------------------------------------------------------

class NearOptimalClassificationTrainer(TrainerBase):
    """Trainer cho bài toán: multi-label near-optimal plan."""

    def construct_training_data(
        self,
        df: pd.DataFrame,
        near_optimal_threshold: float = 1.1,
        default_relative: bool = True,
    ) -> Tuple[List[Any], np.ndarray]:

        """
        near_optimal_threshold:
            - nếu default_relative=False  → dùng như alpha (latency ≤ alpha * best)
            - nếu default_relative=True   → dùng như (1 + τ) trong paper Kepler:
                  (l_d - l_p) * (1 + τ) ≥ (l_d - l_o)
        """

        print("[LOG1] Bắt đầu construct_training_data (Near-Optimal Multi-Label)")

        # làm việc trên copy để không làm bẩn DF gốc
        df = df.copy()
        print("[LOG2] Số dòng exec_df =", len(df))

        # cột param
        col_params = [c for c in df.columns if c.startswith("param")]
        print("[LOG2b] Cột param được phát hiện:", col_params)

        # đảm bảo các cột cơ bản tồn tại
        required_cols = col_params + ["plan_id", "latency_ms"]
        if default_relative:
            required_cols.append("is_default")

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"Thiếu cột {missing} trong exec_latencies.csv. "
                f"Các cột hiện có: {list(df.columns)}"
            )

        # bỏ cột không cần
        df = df.drop(columns=["total_cost"], errors="ignore")

        # preprocessing đặc biệt (hiện tại không dùng)
        self.apply_preprocessing(df)

        # ép kiểu int/float cho các param
        df = cast_df_columns(df, self._predicate_metadata)

        # tạo key tham số
        df["param_key"] = df[col_params].astype(str).agg("####".join, axis=1)
        print("[LOG3] param_key đã tạo, số unique =", df["param_key"].nunique())

        # list các param_key
        unique_keys = df["param_key"].unique()
        num_rows = len(unique_keys)
        print(f"[LOG4] Số bộ tham số distinct = {num_rows}")

        # group theo param_key
        df_group = df.groupby("param_key")

        # chứa feature/target
        features: List[List[Any]] = [[] for _ in col_params]
        targets: List[np.ndarray] = []

        print("[LOG5] Chuẩn bị bắt đầu vòng for duyệt các param_key...")

        for idx, key in enumerate(unique_keys):
            if idx % 2000 == 0:
                print(f"[LOG6] Đang xử lý param_key {idx}/{num_rows}")

            group = df_group.get_group(key)

            # map plan_id -> index theo order trong _plan_ids
           
            latencies = np.full(len(self._plan_ids), np.inf, dtype=np.float32)

            for _, row in group.iterrows():
                pid = int(row["plan_id"])
                if pid not in self._plan_id_to_index:
                    # plan không thuộc plan_cover, bỏ qua
                    continue
                j = self._plan_id_to_index[pid]
                latencies[j] = float(row["latency_ms"])

            # KHÔNG còn bỏ instance chỉ vì thiếu 1 số plan trong cover
            # if np.isinf(latencies).any():
            #     continue

            optimal_latency = float(np.min(latencies))

            # =======================================================
            # NEAR-OPT ĐÚNG THEO PAPER KEPLER (IMPROVEMENT VS DEFAULT)
            # + FALLBACK ĐỂ KHÔNG BỎ HẾT INSTANCE
            # =======================================================
            if default_relative:
                # tìm default latency l_d
                default_rows = group[group["is_default"] == 1]
                if default_rows.empty:
                    # không có default plan cho instance này → bỏ
                    continue

                l_d = float(default_rows["latency_ms"].iloc[0])  # default latency
                l_o = optimal_latency                           # best latency trong cover

                best_improvement = l_d - l_o

                # nếu default đã là tốt nhất hoặc không có cải thiện:
                # vẫn GIỮ instance, gán default là near-opt (single-label)
                if best_improvement <= 0:
                    label = np.zeros(len(self._plan_ids), dtype=np.float32)
                    default_pid = int(default_rows["plan_id"].iloc[0])
                    if default_pid in self._plan_id_to_index:
                        j_def = self._plan_id_to_index[default_pid]
                        label[j_def] = 1.0
                    else:
                        # default không nằm trong plan_cover → bỏ instance này
                        continue

                else:
                    # near_optimal_threshold = (1 + τ)
                    # điều kiện trong paper:
                    #   (l_d - l_p) * (1 + τ) ≥ (l_d - l_o)
                    # ⇔ (l_d - l_p) ≥ (l_d - l_o) / (1 + τ)
                    threshold_improvement = best_improvement / near_optimal_threshold

                    improvements = l_d - latencies  # vector (l_d - l_p)
                    label = (improvements >= threshold_improvement).astype(np.float32)

                    # Nếu vì lý do nào đó tất cả label = 0 → fallback: gán best plan là near-opt
                    if label.sum() == 0:
                        label = np.zeros(len(self._plan_ids), dtype=np.float32)
                        j_best = int(latencies.argmin())
                        label[j_best] = 1.0

            else:
                # GIỮ NGUYÊN CASE CŨ: near-opt theo latency * alpha
                threshold = optimal_latency * near_optimal_threshold
                label = (latencies <= threshold).astype(np.float32)

                # fallback: nếu tất cả =0 → chọn best plan
                if label.sum() == 0:
                    label = np.zeros(len(self._plan_ids), dtype=np.float32)
                    j_best = int(latencies.argmin())
                    label[j_best] = 1.0

            # Không còn check label.sum() == 0 nữa, vì đã fallback phía trên
            targets.append(label)

            # lấy param values từ 1 row bất kỳ trong group
            row0 = group.iloc[0]
            for ci, p in enumerate(col_params):
                features[ci].append(row0[p])

        print("[LOG7] DUYỆT XONG param_key. Đang encode feature/target sang dạng numeric...")

        # Chuyển feature list -> numpy array numeric theo từng kiểu data_type
        X: List[Any] = []
        for ci, f in enumerate(features):
            pred_meta = self._predicate_metadata[ci]
            dtype = pred_meta.get("data_type")

            if dtype == "int":
                arr = np.asarray(f, dtype=np.int64).reshape(-1, 1)
                X.append(arr)

            elif dtype == "float":
                arr = np.asarray(f, dtype=np.float32).reshape(-1, 1)
                X.append(arr)

            elif dtype == "text":
                # Encode text -> id theo vocab trong metadata
                vocab_map = self._text_vocab_maps.get(ci, {})
                if not vocab_map:
                    raise ValueError(
                        f"Không tìm thấy vocab map cho predicate index {ci} (text)."
                    )
                ids: List[int] = []
                for v in f:
                    s = "" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
                    # nếu không có trong vocab, map về 0
                    idx_id = vocab_map.get(s, 0)
                    ids.append(idx_id)

                arr = np.asarray(ids, dtype=np.int64).reshape(-1, 1)
                X.append(arr)

            else:
                raise ValueError(f"Unsupported data_type for features: {dtype}")

        if len(targets) > 0:
            Y = np.vstack(targets).astype(np.float32)
        else:
            Y = np.zeros((0, len(self._plan_ids)), dtype=np.float32)

        print(
            "[LOG8] Feature summary:",
            [
                f"idx={i}, type={type(x)}, "
                f"shape={getattr(x, 'shape', None)}, "
                f"dtype={getattr(x, 'dtype', None)}"
                for i, x in enumerate(X)
            ],
        )
        print("[LOG9] Target shape =", Y.shape, ", dtype =", Y.dtype)
        print("[LOG10] Hoàn tất construct_training_data.")

        return X, Y
