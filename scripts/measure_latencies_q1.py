#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from typing import Any, Dict, List

import psycopg2

QUERY_ID = "q1_0"
NAME_DELIMITER = "####"

DB_CONF = {
    "dbname": "stack",
    "user": "postgres",
    "password": "kelper",
    "host": "localhost",
    "port": 5432,
}

JSON = Any


def load_params(params_path: str) -> Dict[str, Any]:
    with open(params_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if QUERY_ID not in data:
        raise ValueError(f"Không thấy {QUERY_ID} trong file params {params_path}")
    return data[QUERY_ID]


def load_plans(plans_path: str) -> List[Dict[str, Any]]:
    with open(plans_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if QUERY_ID not in data:
        raise ValueError(f"Không thấy {QUERY_ID} trong file plans {plans_path}")
    return data[QUERY_ID]


def load_debug_infos(debug_infos_path: str) -> Dict[str, Any]:
    with open(debug_infos_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if QUERY_ID not in data:
        raise ValueError(f"Không thấy {QUERY_ID} trong file debug_infos {debug_infos_path}")
    return data[QUERY_ID]


def apply_hints(query_sql: str, hints_str: str) -> str:
    hints_str = hints_str.strip()
    if not hints_str:
        return query_sql
    if hints_str.startswith("/*"):
        return f"{hints_str} {query_sql}"
    return f"/*+ {hints_str} */ {query_sql}"


def substitute_params(query_sql: str, params: List[Any]) -> str:
    q = query_sql
    for i, p in enumerate(params):
        q = q.replace(f"@param{i}", str(p))
    return q


def parse_explain_json(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, str):
        data = json.loads(raw)
    else:
        data = raw

    if isinstance(data, list) and data:
        obj = data[0]
    else:
        obj = data

    if not isinstance(obj, dict):
        return {}

    return obj


def get_latency_ms_from_explain_obj(obj: Dict[str, Any]) -> float | None:
    if "Execution Time" in obj:
        return float(obj["Execution Time"])

    plan = obj.get("Plan")
    if isinstance(plan, dict) and "Actual Total Time" in plan:
        return float(plan["Actual Total Time"])

    return None


def measure_latencies_q1(
    params_path: str,
    plans_path: str,
    debug_infos_path: str,
    out_csv: str,
    repeats: int = 1,
    max_instances: int | None = None,
    statement_timeout_ms: int = 45000,
    max_plans_per_instance: int | None = None,
    sample_seed: int = 42,
) -> None:
    # ----- load dữ liệu -----
    entry = load_params(params_path)
    query_sql: str = entry["query"]
    params_list: List[List[Any]] = entry["params"]

    if max_instances is not None:
        params_list = params_list[: max_instances]

    plans = load_plans(plans_path)
    debug_infos = load_debug_infos(debug_infos_path)

    # map: plan_idx -> is_default
    idx_is_default: Dict[int, bool] = {}
    for idx, plan in enumerate(plans):
        hint = plan.get("hints", "")
        info = debug_infos.get(hint, {})
        idx_is_default[idx] = (info.get("source") == "default")

    # tìm default_idx và non_default_indices
    default_indices: List[int] = [i for i, is_def in idx_is_default.items() if is_def]
    if not default_indices:
        raise RuntimeError("Không tìm thấy plan nào có source=default trong debug_infos!")
    if len(default_indices) > 1:
        print(f"⚠ Cảnh báo: có {len(default_indices)} default plans, sẽ dùng plan_idx = {default_indices[0]}")

    default_idx = default_indices[0]
    non_default_indices: List[int] = [i for i in range(len(plans)) if i != default_idx]

    # nếu không set max_plans_per_instance → đo tất cả (default + toàn bộ non-default)
    if max_plans_per_instance is None:
        num_plans_per_instance = 1 + len(non_default_indices)
    else:
        # luôn giữ 1 default, còn lại là non-default sample
        num_non_default_to_sample = max(0, max_plans_per_instance - 1)
        num_non_default_to_sample = min(num_non_default_to_sample, len(non_default_indices))
        num_plans_per_instance = 1 + num_non_default_to_sample

    num_instances = len(params_list)
    print(f"▶ Đo latency cho {QUERY_ID}")
    print(f" - Số instance: {num_instances}")
    print(f" - Số plan unique: {len(plans)}")
    print(f" - Plan default idx: {default_idx}")
    print(f" - Số non-default: {len(non_default_indices)}")
    print(f" - Số plan đo mỗi instance: {num_plans_per_instance}")

    total_rows_est = num_instances * num_plans_per_instance * repeats
    print(f" - Tổng bản ghi dự kiến: {total_rows_est}")

    # ----- connect PostgreSQL -----
    print("🔌 Connecting to PostgreSQL...")
    conn = psycopg2.connect(
        dbname=DB_CONF["dbname"],
        user=DB_CONF["user"],
        password=DB_CONF["password"],
        host=DB_CONF["host"],
        port=DB_CONF["port"],
    )
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute(f"SET statement_timeout TO {statement_timeout_ms};")

    random.seed(sample_seed)

    # ----- chuẩn bị output CSV -----
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            ["instance_idx", "param_key", "plan_idx", "latency_ms", "is_default", "error"]
        )

        for inst_idx, params in enumerate(params_list):
            param_key = NAME_DELIMITER.join(str(p) for p in params)
            print(
                f"  - Instance {inst_idx+1}/{num_instances}: {param_key}"
            )

            # luôn đo default plan
            plan_indices_to_run: List[int] = [default_idx]

            # sample thêm non-default
            if max_plans_per_instance is None or len(non_default_indices) <= (num_plans_per_instance - 1):
                sampled_non_default = non_default_indices
            else:
                sampled_non_default = random.sample(
                    non_default_indices,
                    num_plans_per_instance - 1,
                )

            plan_indices_to_run.extend(sampled_non_default)

            for plan_idx in plan_indices_to_run:
                hints_str = plans[plan_idx].get("hints", "")
                is_default = 1 if plan_idx == default_idx else 0

                hinted_query = apply_hints(query_sql, hints_str)
                query_with_params = substitute_params(hinted_query, params)

                analyze_sql = f"EXPLAIN (ANALYZE, FORMAT JSON) {query_with_params}"

                for _ in range(repeats):
                    try:
                        cur.execute(analyze_sql)
                        raw = cur.fetchone()[0]
                        explain_obj = parse_explain_json(raw)
                        latency_ms = get_latency_ms_from_explain_obj(explain_obj)

                        if latency_ms is None:
                            writer.writerow(
                                [
                                    inst_idx,
                                    param_key,
                                    plan_idx,
                                    "",
                                    is_default,
                                    "no_latency_in_explain",
                                ]
                            )
                        else:
                            writer.writerow(
                                [
                                    inst_idx,
                                    param_key,
                                    plan_idx,
                                    round(latency_ms, 3),
                                    is_default,
                                    "",
                                ]
                            )
                    except Exception as e:
                        writer.writerow(
                            [
                                inst_idx,
                                param_key,
                                plan_idx,
                                "",
                                is_default,
                                f"exception: {e}",
                            ]
                        )

    cur.close()
    conn.close()
    print(f"✅ Đã lưu file latency CSV: {out_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Đo latency cho q1_0 bằng EXPLAIN (ANALYZE, FORMAT JSON) với sampling plans."
    )
    parser.add_argument(
        "--params-path",
        default="data/stack_params/q1_0-178214.json",
    )
    parser.add_argument(
        "--plans-path",
        default="execution_data/candidate_plans/rce_q1_0_plans.json",
    )
    parser.add_argument(
        "--debug-infos-path",
        default="execution_data/candidate_plans/rce_q1_0_plans_debug_infos.json",
    )
    parser.add_argument(
        "--out",
        default="artifacts/exec_latencies_q1.csv",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--statement-timeout-ms",
        type=int,
        default=45000,
    )
    parser.add_argument(
        "--max-plans-per-instance",
        type=int,
        default=21,
        help="Tổng số plan đo cho mỗi instance (gồm 1 default + K non-default).",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed random cho việc chọn plan non-default.",
    )

    args = parser.parse_args()

    measure_latencies_q1(
        params_path=args.params_path,
        plans_path=args.plans_path,
        debug_infos_path=args.debug_infos_path,
        out_csv=args.out,
        repeats=args.repeats,
        max_instances=args.max_instances,
        statement_timeout_ms=args.statement_timeout_ms,
        max_plans_per_instance=args.max_plans_per_instance,
        sample_seed=args.sample_seed,
    )


if __name__ == "__main__":
    main()
