# sngp_pipeline/pipeline_data.py
from __future__ import annotations

import copy
import enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

JSON = Any

# Tên key & delimiter giống Kepler gốc
_NAME_DELIMITER = "####"

_DEFAULT = "default"
_EXPLAINS = "explains"
_TOTAL_COST = "total_cost"


# ----------------------------------------------------------------------
#  Enum estimator: lấy min / max / median trong list thời gian chạy
# ----------------------------------------------------------------------
class LatencyEstimator(str, enum.Enum):
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"


ESTIMATOR_MAP = {
    LatencyEstimator.MIN: np.min,
    LatencyEstimator.MAX: np.max,
    LatencyEstimator.MEDIAN: np.median,
}


# ----------------------------------------------------------------------
#  PlannedQuery: 1 query cụ thể (template + plan + params)
# ----------------------------------------------------------------------
@dataclass
class PlannedQuery:
    """Một instance query với plan đã chọn.

    Attributes:
      query_id: id template (ví dụ 'q1_0').
      plan_id: id của plan muốn chạy; None nghĩa là dùng default.
      parameters: list giá trị param (['stackoverflow', 'iron-router', ...])
    """
    query_id: str
    plan_id: Optional[int]
    parameters: List[str]


def _fetch_plan_id(
    default_plan_id: int,
    plan_id: Optional[int],
    max_possible_plan_id: int,
) -> int:
    """Resolve plan_id = None → default, và validate."""
    if plan_id is None:
        plan_id = default_plan_id

    if plan_id < 0 or plan_id > max_possible_plan_id:
        raise ValueError(
            f"Provided plan id ({plan_id}) does not refer to a recognized plan."
        )
    return plan_id


def _is_plan_skipped(stats: Any, plan_id: int) -> bool:
    """Kiểm tra plan này có bị 'skipped' trong execution_data hay không."""
    plan_results = stats["results"][plan_id]
    return any("skipped" in elem for elem in plan_results)


def _is_plan_cover_plan_skipped(
    stats: Any,
    plan_cover: List[int],
) -> Optional[int]:
    """Trả về plan_id trong plan_cover bị skipped (nếu có)."""
    for pid in plan_cover:
        if _is_plan_skipped(stats, pid):
            return pid
    return None


# ----------------------------------------------------------------------
#  DatabaseSimulator: giả lập DB trên execution_data JSON của tác giả
# ----------------------------------------------------------------------
class DatabaseSimulator:
    """Simulates database interactions backed by Kepler Stack execution data.

    - query_execution_data: dict từ JSON stack_qX_Y.json
    - query_execution_metadata: dict từ stack_qX_Y_metadata.json
      (ít nhất phải có field "plan_cover")
    """

    def __init__(
        self,
        query_execution_data: Any,
        query_execution_metadata: Any,
        estimator: LatencyEstimator,
        query_explain_data: Optional[Any] = None,
    ):
        if len(query_execution_data) != 1:
            raise ValueError("Unexpected data format for query_execution_data.")
        if len(query_execution_metadata) != 1:
            raise ValueError("Unexpected data format for query_execution_metadata.")
        if query_explain_data is not None and len(query_explain_data) != 1:
            raise ValueError("Unexpected data format for query_explain_data.")

        # Lấy query_id (ví dụ 'q1_0')
        self.query_id = next(iter(query_execution_data))

        if self.query_id not in query_execution_metadata:
            raise ValueError(
                f"Query id mismatch between data arguments. Found {self.query_id} in "
                f"query_execution_data and {next(iter(query_execution_metadata))} in "
                "query_execution_metadata."
            )

        if query_explain_data is not None and self.query_id not in query_explain_data:
            raise ValueError(
                f"Query id mismatch between data arguments. Found {self.query_id} in "
                f"query_execution_data and {next(iter(query_explain_data))} in "
                "query_explain_data."
            )

        data_mapping = query_execution_data[self.query_id]
        meta = query_execution_metadata[self.query_id]
        plan_cover = meta.get("plan_cover", [])

        self._table: Dict[Tuple[str, ...], Dict[str, Any]] = {}

        # Duyệt từng key param trong execution_data
        for parameters_as_key, stats in data_mapping.items():
            # Không cho tồn tại key "explains" ở execution_data
            if _EXPLAINS in stats:
                raise ValueError(
                    f"Execution data contains key '{_EXPLAINS}', which "
                    "is reserved for query_explain_data."
                )

            # Nếu có default_timed_out ⇒ bỏ qua binding này
            if "default_timed_out" in stats.get("results", {}):
                continue

            # Các parameter value được ghép bằng "####"
            params_as_tuple = tuple(parameters_as_key.rsplit(_NAME_DELIMITER))

            # Kiểm tra default plan không bị skipped
            default_pid = stats[_DEFAULT]
            if _is_plan_skipped(stats=stats, plan_id=default_pid):
                raise ValueError(
                    f"Default plan {default_pid} skipped for params: {params_as_tuple}"
                )

            # Kiểm tra các plan trong plan_cover không bị skipped
            bad_pid = _is_plan_cover_plan_skipped(stats=stats, plan_cover=plan_cover)
            if bad_pid is not None:
                raise ValueError(
                    f"Plan id {bad_pid} from plan_cover {plan_cover} skipped "
                    f"for params: {params_as_tuple}"
                )

            # Lưu stats vào bảng giả lập
            self._table[params_as_tuple] = copy.deepcopy(stats)

            # Nếu có query_explain_data thì merge thêm _EXPLAINS
            if query_explain_data is not None:
                if parameters_as_key not in query_explain_data[self.query_id]:
                    raise ValueError(
                        "query_explain_data must contain all parameter keys "
                        f"found in query_execution_data, but {parameters_as_key} "
                        "not found."
                    )
                explains = query_explain_data[self.query_id][parameters_as_key]
                self._table[params_as_tuple][_EXPLAINS] = copy.deepcopy(
                    explains["results"]
                )

        self._estimator = estimator
        self.execution_count = 0
        self.execution_cost_ms = 0.0

    # ---------------- internal helper ----------------
    def _fetch_entry(self, planned_query: PlannedQuery) -> Dict[str, Any]:
        if self.query_id != planned_query.query_id:
            raise ValueError(
                f"DatabaseSimulator is for query template {self.query_id} but "
                f"planned_query has template {planned_query.query_id}"
            )

        if len(planned_query.parameters) != len(next(iter(self._table))):
            raise ValueError("All parameter bindings must be provided.")

        key = tuple(planned_query.parameters)
        if key not in self._table:
            raise ValueError(
                f"Out-of-universe query with plan {planned_query.plan_id} and "
                f"parameters: {planned_query.parameters}"
            )

        return self._table[key]

    # ---------------- public API ----------------
    def get_plan_id(self, planned_query: PlannedQuery) -> int:
        entry = self._fetch_entry(planned_query)
        pid = _fetch_plan_id(
            default_plan_id=entry[_DEFAULT],
            plan_id=planned_query.plan_id,
            max_possible_plan_id=len(entry["results"]) - 1,
        )
        return pid

    def execute_explain(self, planned_query: PlannedQuery) -> Tuple[float, int]:
        """Trả về (total_cost, plan_id) từ EXPLAIN."""
        entry = self._fetch_entry(planned_query)
        if _EXPLAINS not in entry:
            raise ValueError(
                "Called execute_explain without providing query_explain_data "
                "in the init()."
            )

        pid = _fetch_plan_id(
            default_plan_id=entry[_DEFAULT],
            plan_id=planned_query.plan_id,
            max_possible_plan_id=len(entry["results"]) - 1,
        )

        total_cost = entry[_EXPLAINS][pid][0][_TOTAL_COST]
        return float(total_cost), pid

    def execute_timed(self, planned_query: PlannedQuery) -> Tuple[float, bool]:
        """Simulate execute query, trả về (latency_ms, is_default)."""
        entry = self._fetch_entry(planned_query)

        pid = _fetch_plan_id(
            default_plan_id=entry[_DEFAULT],
            plan_id=planned_query.plan_id,
            max_possible_plan_id=len(entry["results"]) - 1,
        )

        if _is_plan_skipped(entry, pid):
            raise ValueError(
                "Cannot execute query. No execution data was provided for plan "
                f"{pid} with parameter binding: {planned_query.parameters}"
            )

        plan_results = entry["results"][pid]
        is_default = (entry[_DEFAULT] == pid)

        # Nếu có timeout, lấy max duration_ms
        if any("timed_out" in d for d in plan_results):
            latency = np.max([next(iter(d.values())) for d in plan_results])
        else:
            estimator_fn = ESTIMATOR_MAP[self._estimator]
            latency = estimator_fn([d["duration_ms"] for d in plan_results])

        self.execution_count += 1
        self.execution_cost_ms += float(latency)

        return float(latency), bool(is_default)


# ----------------------------------------------------------------------
#  DatabaseClient: wrapper trả về DataFrame giống exec_latencies.csv
# ----------------------------------------------------------------------
class DatabaseClient:
    """Wrapper để gọi DatabaseSimulator nhiều lần và trả về DataFrame."""

    def __init__(self, database: DatabaseSimulator):
        self._database = database

    def execute_timed_batch(
        self,
        planned_queries: List[PlannedQuery],
        get_total_cost: bool = False,
    ) -> pd.DataFrame:
        """Chạy batch PlannedQuery, trả về DataFrame.

        Columns:
          param0, param1, ..., plan_id, total_cost, latency_ms, is_default
        """
        if not planned_queries:
            raise ValueError("Cannot execute empty batch.")

        data_rows = []
        for pq in planned_queries:
            latency, is_default = self._database.execute_timed(pq)

            if get_total_cost:
                total_cost, plan_id = self._database.execute_explain(pq)
            else:
                total_cost = -1.0
                plan_id = self._database.get_plan_id(pq)

            # ghép params + [plan_id, total_cost, latency, is_default]
            data_rows.append(
                list(pq.parameters)
                + [int(plan_id), float(total_cost), float(latency), bool(is_default)]
            )

        first_q = planned_queries[0]
        param_cols = [f"param{i}" for i in range(len(first_q.parameters))]
        cols = param_cols + ["plan_id", _TOTAL_COST, "latency_ms", "is_default"]

        df = pd.DataFrame(data_rows, columns=cols)
        return df
