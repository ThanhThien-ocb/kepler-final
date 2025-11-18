# scripts/01_generate_latencies.py
#
# Convert Kepler Stack execution_data (stack_qX_Y.json) + metadata
# thành:
#   - data/exec_latencies.csv
#   - data/metadata.json
#   - data/plan_cover.json
#
# Dựa đúng theo cấu trúc DatabaseSimulator trong Kepler:
#   query_execution_data = {
#     "q1_0": {
#       "param1####param2...": {
#         "default": <int>,
#         "results": [
#           [ {"duration_ms": ...}, {"duration_ms": ...}, ... ],  # plan 0
#           [ {"timed_out": ...}, {"duration_ms": ...}, ... ],   # plan 1
#           ...
#         ],
#         "execution_order": ...,
#         "rows": ...,
#         "timeout_ms": ...
#       },
#       ...
#     }
#   }

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

JSON = Any

_NAME_DELIMITER = "####"
_DEFAULT = "default"


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
        if not txt:
            raise RuntimeError(f"File rỗng: {path}")
        return json.loads(txt)


def build_metadata_from_example(
    query_id: str,
    example_key: str,
) -> Dict[str, Any]:
    """Tạo metadata với predicates = param0, param1, ... (kiểu text)."""
    parts = example_key.rsplit(_NAME_DELIMITER)
    predicates: List[Dict[str, Any]] = []

    for i, _ in enumerate(parts):
        predicates.append(
            {
                "name": f"param{i}",
                "data_type": "text",  # vì trong exec JSON chỉ có string
            }
        )

    metadata: Dict[str, Any] = {
        "query_id": query_id,
        "predicates": predicates,
    }
    return metadata


def compute_latency_ms(plan_results: List[Dict[str, Any]]) -> float:
    """Tính latency giống DatabaseSimulator.execute_timed (estimator=MEDIAN).

    - Nếu có 'timed_out' → latency = max(all observed durations/timed_out)
    - Ngược lại → median(duration_ms) trong plan_results
    """
    if any("timed_out" in d for d in plan_results):
        # Mỗi dict trong plan_results có 1 key duy nhất
        vals = [next(iter(d.values())) for d in plan_results]
        return float(np.max(vals))

    durations = [d["duration_ms"] for d in plan_results]
    return float(np.median(durations))


def convert_to_latencies_df(
    exec_data: Dict[str, Any],
    meta_data: Dict[str, Any],
    query_id: str,
) -> (pd.DataFrame, Dict[str, Any], List[int]):
    """Flatten execution_data của 1 query_id thành DataFrame exec_latencies.

    Returns:
      - DataFrame với cột param0.., plan_id, total_cost, latency_ms, is_default
      - metadata (query_id + predicates)
      - plan_cover (list[int])
    """
    if query_id not in exec_data:
        # nếu user truyền q1_0 nhưng file là q1, fallback key đầu tiên
        query_id = next(iter(exec_data.keys()))
        print(f"query_id không có trong execution_data, fallback sang '{query_id}'")

    if query_id not in meta_data:
        raise RuntimeError(
            f"Không tìm thấy query_id={query_id} trong execution metadata."
        )

    data_mapping: Dict[str, Any] = exec_data[query_id]
    meta_q: Dict[str, Any] = meta_data[query_id]
    plan_cover: List[int] = meta_q.get("plan_cover", None)

    # Lấy 1 ví dụ key bất kỳ để biết số param
    example_key = next(iter(data_mapping.keys()))
    metadata = build_metadata_from_example(query_id, example_key)
    num_params = len(metadata["predicates"])

    # Nếu metadata không có plan_cover, lấy toàn bộ plan_id từ 1 stats bất kỳ
    if plan_cover is None:
        example_stats = data_mapping[example_key]
        num_plans = len(example_stats["results"])
        plan_cover = list(range(num_plans))
        print(f"Không thấy 'plan_cover' trong metadata, dùng 0..{num_plans-1}")

    rows: List[Dict[str, Any]] = []

    for parameters_as_key, stats in data_mapping.items():
        # stats là dict: {'default': int, 'results': [...], 'execution_order':..., 'rows':..., 'timeout_ms':...}
        if "results" not in stats or _DEFAULT not in stats:
            continue

        # Trong 1 số exec data có "default_timed_out" trong stats["results"]
        if isinstance(stats["results"], dict) and "default_timed_out" in stats["results"]:
            # dạng hiếm, bỏ qua luôn
            continue

        # param values
        parts = parameters_as_key.rsplit(_NAME_DELIMITER)
        if len(parts) != num_params:
            # không khớp số lượng param, bỏ qua cho an toàn
            continue

        default_plan_id = stats[_DEFAULT]
        results_list = stats["results"]  # list[size = num_plans], mỗi phần tử là list[dict]

        for plan_id in plan_cover:
            if plan_id < 0 or plan_id >= len(results_list):
                continue

            plan_results = results_list[plan_id]
            if not isinstance(plan_results, list) or not plan_results:
                continue

            # nếu có 'skipped' trong bất kỳ dict → bỏ luôn (giống _is_plan_skipped)
            if any("skipped" in d for d in plan_results):
                continue

            latency_ms = compute_latency_ms(plan_results)
            is_default = (plan_id == default_plan_id)

            row: Dict[str, Any] = {}
            for i, val in enumerate(parts):
                row[f"param{i}"] = val

            row["plan_id"] = int(plan_id)
            row["total_cost"] = -1  # Không dùng EXPLAIN, placeholder
            row["latency_ms"] = latency_ms
            row["is_default"] = bool(is_default)

            rows.append(row)

    if not rows:
        raise RuntimeError("Không thu được hàng nào cho exec_latencies.csv")

    df = pd.DataFrame(rows)
    print(f"DataFrame exec_latencies: {df.shape[0]} dòng, {df.shape[1]} cột")
    print("  Các cột:", list(df.columns))

    return df, metadata, plan_cover


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert Kepler Stack execution_output JSON -> "
            "exec_latencies.csv, metadata.json, plan_cover.json."
        )
    )
    parser.add_argument(
        "--root-dir",
        default="data/stack_parametric_query_dataset",
        help="Thư mục gốc dataset (chứa execution_data/).",
    )
    parser.add_argument(
        "--query-id",
        default="q1_0",
        help="Query ID muốn dùng (vd: q1_0, q4_0, ...).",
    )
    parser.add_argument(
        "--out-lat",
        default="data/exec_latencies.csv",
        help="Output exec_latencies.csv trong project kepler-sngp.",
    )
    parser.add_argument(
        "--out-metadata",
        default="data/metadata.json",
        help="Output metadata.json (predicates + query_id).",
    )
    parser.add_argument(
        "--out-plan-cover",
        default="data/plan_cover.json",
        help="Output plan_cover.json (list plan_id).",
    )

    args = parser.parse_args()

    root = args.root_dir
    qid = args.query_id

    execution_data_path = os.path.join(
        root,
        "execution_data",
        "results",
        qid,
        "execution_output",
        f"stack_{qid}.json",
    )
    execution_metadata_path = os.path.join(
        root,
        "execution_data",
        "results",
        qid,
        "execution_output",
        f"stack_{qid}_metadata.json",
    )

    print("=== BẮT ĐẦU CONVERT DATASET TÁC GIẢ (format DatabaseSimulator) ===")
    print(f"ROOT_DIR           = {root}")
    print(f"QUERY_ID           = {qid}")
    print(f"EXECUTION_DATA     = {execution_data_path}")
    print(f"EXECUTION_METADATA = {execution_metadata_path}")
    print("")

    exec_data = load_json(execution_data_path)
    meta_data = load_json(execution_metadata_path)

    df, metadata, plan_cover = convert_to_latencies_df(exec_data, meta_data, qid)

    # 1) exec_latencies.csv
    os.makedirs(os.path.dirname(args.out_lat), exist_ok=True)
    df.to_csv(args.out_lat, index=False)
    print(f"1/Đã lưu exec_latencies.csv → {args.out_lat}")

    # 2) metadata.json
    os.makedirs(os.path.dirname(args.out_metadata), exist_ok=True)
    with open(args.out_metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"2/Đã lưu metadata.json → {args.out_metadata}")

    # 3) plan_cover.json
    os.makedirs(os.path.dirname(args.out_plan_cover), exist_ok=True)
    with open(args.out_plan_cover, "w", encoding="utf-8") as f:
        json.dump(plan_cover, f, indent=2)
    print(f"3/ Đã lưu plan_cover.json ({len(plan_cover)} plans) → {args.out_plan_cover}")

    print("=== HOÀN TẤT CONVERT DATASET TÁC GIẢ ===")


if __name__ == "__main__":
    main()
