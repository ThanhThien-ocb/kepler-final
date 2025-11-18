#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FULL Row Count Evolution (RCE) cho query q1_0 — chuẩn Kepler 2023

Sinh candidate plans đầy đủ:
  - Random Scan Method (SeqScan, IndexScan, BitmapScan, IndexOnlyScan)
  - Random Join Method (NestLoop, HashJoin, MergeJoin)
  - Random Join Order bằng Leading(...)
  - Cardinality perturbation mạnh (multi-dimensional)
  - Dedup plan signature chuẩn pg_hint_plan
  - Mapping param_key -> plan_id (index)

Nhận dạng giống Kepler:
    rce_q1_0_plans.json
    rce_q1_0_plans_plan_indices.json
    rce_q1_0_plans_debug_infos.json
    rce_q1_0_failures.json

Command:
python3 scripts/rce_generate_q1_0.py \
  --params-path data/stack_params/q1_0-178214.json \
  --output-dir execution_data/candidate_plans \
  --db-name stack \
  --db-user postgres \
  --db-password kelper \
  --db-host localhost \
  --max-instances 100 \
  --max-plans-per-instance 50
"""

from __future__ import annotations
import os, sys, json, random, math
from typing import Dict, Any, List

CURRENT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

KEPLER_DEMO = os.path.expanduser("~/kepler-demo")
if os.path.isdir(KEPLER_DEMO) and KEPLER_DEMO not in sys.path:
    sys.path.insert(0, KEPLER_DEMO)

from kepler.training_data_collection_pipeline import pg_plan_hint_extractor
from kepler.training_data_collection_pipeline import query_utils
from kepler.training_data_collection_pipeline import query_text_utils
from kepler.training_data_collection_pipeline import main_utils

JSON = Any
QUERY_ID = "q1_0"
NAME_DELIMITER = "####"

SCAN_METHODS = ["SeqScan", "IndexScan", "BitmapScan", "IndexOnlyScan"]
JOIN_METHODS = ["NestLoop", "HashJoin", "MergeJoin"]


# --------------------------------------------------------------------
# Perturb cardinality (Kepler-style)
# --------------------------------------------------------------------
def perturb_cardinality(row_counts: Dict[str, int], m=10.0, base=5.0):
    r_new = {}
    for jt, rc in row_counts.items():
        w = max(rc, 1)
        logw = math.log(w, base)
        e_l = -min(logw, m)
        e_u = e_l + 2 * m
        exp = random.uniform(e_l, e_u)
        f = base ** exp
        r_new[jt] = max(1, int(w * f))
    return r_new


def rows_hint_from_map(rmap: Dict[str, int]) -> List[str]:
    return [f"Rows({jt} #{rc})" for jt, rc in rmap.items()]


# --------------------------------------------------------------------
# Random scan + join + join order (Leading)
# --------------------------------------------------------------------
def random_scan_hints(tables: List[str]):
    return [f"{random.choice(SCAN_METHODS)}({t})" for t in tables]


def random_join_hints(jtrees: List[str]):
    return [f"{random.choice(JOIN_METHODS)}({jt})" for jt in jtrees]


def random_leading(tables: List[str]):
    tb = list(tables)
    random.shuffle(tb)
    return f"Leading({' '.join(tb)})"


def assemble_hint(scan, join, lead, rows):
    allp = scan + join + [lead] + rows
    return f"/*+ {' '.join(allp)} */"


# --------------------------------------------------------------------
# Generate plans for ONE instance
# --------------------------------------------------------------------
def generate_instance_candidates(
    query_sql: str,
    params: List[Any],
    manager,
    max_plans=50
):
    # Default plan
    base_hint, plan = pg_plan_hint_extractor.get_single_query_hints_with_plan(
        query_manager=manager,
        query=query_sql,
        params=params,
    )

    row_counts = pg_plan_hint_extractor.extract_row_counts(plan["Plan"])

    # Base tables
    base_tables = set()
    for jt in row_counts.keys():
        for t in jt.split():
            base_tables.add(t)
    base_tables = list(base_tables)

    join_trees = list(row_counts.keys())

    cand = {base_hint: {"source": "default"}}

    for _ in range(max_plans * 5):
        if len(cand) >= max_plans:
            break

        rmap = perturb_cardinality(row_counts)
        rows_h = rows_hint_from_map(rmap)
        scan_h = random_scan_hints(base_tables)
        join_h = random_join_hints(join_trees)
        lead_h = random_leading(base_tables)

        full_hint = assemble_hint(scan_h, join_h, lead_h, rows_h)
        hinted_query = query_text_utils.get_hinted_query(query_sql, full_hint)

        new_hint, _ = pg_plan_hint_extractor.get_single_query_hints_with_plan(
            query_manager=manager,
            query=hinted_query,
            params=params,
        )

        if new_hint not in cand:
            cand[new_hint] = {
                "source": "rce_full",
                "debug": {
                    "scan": scan_h,
                    "join": join_h,
                    "lead": lead_h,
                    "rows": rmap,
                },
            }

    return cand


# --------------------------------------------------------------------
# MAIN RCE on whole dataset
# --------------------------------------------------------------------
def run_rce(
    params_path: str,
    output_dir: str,
    dbname: str,
    user: str,
    password: str,
    host: str,
    max_instances,
    max_plans_per_instance,
):
    with open(params_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entry = data[QUERY_ID]
    query_sql = entry["query"]
    params_list = entry["params"]

    if max_instances is not None:
        params_list = params_list[:max_instances]

    manager = query_utils.QueryManager(
        query_utils.DatabaseConfiguration(
            dbname=dbname, user=user, password=password, host=host
        )
    )

    acc = main_utils.HintAccumulator()
    acc.query_id_to_plan_hints[QUERY_ID] = []
    acc.query_id_to_params_plan_indices[QUERY_ID] = {}
    acc.query_id_to_debug_infos[QUERY_ID] = {}
    acc.combined_failure_counts[QUERY_ID] = {}

    hint_to_idx = {}

    for i, params in enumerate(params_list):
        param_key = NAME_DELIMITER.join(str(p) for p in params)
        print(f"▶ Instance {i+1}/{len(params_list)}: {param_key}")

        cands = generate_instance_candidates(
            query_sql,
            params,
            manager,
            max_plans=max_plans_per_instance,
        )

        for hint_str, dbg in cands.items():
            if hint_str not in hint_to_idx:
                idx = len(acc.query_id_to_plan_hints[QUERY_ID])
                acc.query_id_to_plan_hints[QUERY_ID].append({"hints": hint_str})
                hint_to_idx[hint_str] = idx
                acc.query_id_to_debug_infos[QUERY_ID][hint_str] = dbg
                acc.combined_failure_counts[QUERY_ID][hint_str] = 0
            else:
                idx = hint_to_idx[hint_str]

            acc.query_id_to_params_plan_indices[QUERY_ID].setdefault(param_key, [])
            if idx not in acc.query_id_to_params_plan_indices[QUERY_ID][param_key]:
                acc.query_id_to_params_plan_indices[QUERY_ID][param_key].append(idx)

    os.makedirs(output_dir, exist_ok=True)
    acc.save(
        output_dir,
        "rce_q1_0_plans.json",
        "rce_q1_0_failures.json",
        "_plan_indices.json",
    )
    print(" Hoàn thành RCE FULL Kepler")


# --------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Full RCE for q1_0 (Kepler style)")
    parser.add_argument("--params-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--db-name", required=True)
    parser.add_argument("--db-user", required=True)
    parser.add_argument("--db-password", required=True)
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--max-plans-per-instance", type=int, default=50)
    args = parser.parse_args()

    run_rce(
        params_path=args.params_path,
        output_dir=args.output_dir,
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_password,
        host=args.db_host,
        max_instances=args.max_instances,
        max_plans_per_instance=args.max_plans_per_instance,
    )


if __name__ == "__main__":
    main()
