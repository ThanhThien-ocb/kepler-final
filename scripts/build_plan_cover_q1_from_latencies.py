#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build plan_cover cho q1_0 từ file latency CSV.

Input:
  - --lat: artifacts/exec_latencies_q1.csv
    (format: instance_idx,param_key,plan_idx,latency_ms,is_default,error)

Output:
  - --out: data/plan_cover_q1.json
    (format: [plan_id_0, plan_id_1, ...])

Ý tưởng:
  - Với mỗi instance (param_key), lấy latency tốt nhất (oracle).
  - Một plan p "cover" instance i nếu:
        latency(p, i) <= alpha * best_latency(i)
    với alpha ~ 1.1 (mặc định).
  - Dùng greedy set cover: mỗi bước chọn plan cover thêm được nhiều instance
    chưa cover nhất.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, Set

import pandas as pd


def build_plan_cover(
    lat_path: str,
    out_path: str,
    alpha: float = 1.1,
    max_plans: int | None = None,
) -> None:
    print(f"Đọc latency CSV từ: {lat_path}")
    df = pd.read_csv(lat_path)

    # Lọc các dòng OK: có latency_ms và không lỗi
    df = df[df["latency_ms"].notna()]
    if "error" in df.columns:
        df = df[(df["error"].isna()) | (df["error"] == "")]

    # Nếu đo lặp nhiều lần, lấy median latency cho mỗi (instance, plan)
    agg = (
        df.groupby(["param_key", "plan_idx"], as_index=False)["latency_ms"]
        .median()
        .rename(columns={"plan_idx": "plan_id"})
    )

    # Latency tốt nhất (oracle) cho mỗi instance
    best_latency = (
        agg.groupby("param_key")["latency_ms"].min().to_dict()
    )  # type: Dict[str, float]

    print(f"Số instance (param_key): {len(best_latency)}")
    print(f"Số (instance, plan) quan sát được: {len(agg)}")

    # Tập instance mà mỗi plan cover (near-optimal)
    plan_to_instances: Dict[int, Set[str]] = {}

    for _, row in agg.iterrows():
        param_key = row["param_key"]
        plan_id = int(row["plan_id"])
        lat = float(row["latency_ms"])
        if lat <= alpha * best_latency[param_key]:
            plan_to_instances.setdefault(plan_id, set()).add(param_key)

    universe: Set[str] = set(best_latency.keys())
    covered: Set[str] = set()
    plan_cover: list[int] = []

    print(f"Bắt đầu greedy set cover với alpha = {alpha}")
    step = 0
    while covered != universe:
        best_plan = None
        best_gain = 0

        for plan_id, insts in plan_to_instances.items():
            gain = len(insts - covered)
            if gain > best_gain:
                best_gain = gain
                best_plan = plan_id

        if best_plan is None or best_gain == 0:
            print("⚠ Không còn plan nào giúp cover thêm instance mới → dừng.")
            break

        plan_cover.append(int(best_plan))
        newly_covered = plan_to_instances[best_plan] - covered
        covered |= plan_to_instances[best_plan]

        step += 1
        print(
            f"  - Bước {step}: chọn plan {best_plan}, "
            f"thêm {len(newly_covered)} instance, "
            f"tổng cover = {len(covered)}/{len(universe)}"
        )

        if max_plans is not None and len(plan_cover) >= max_plans:
            print(f"Đã đạt max_plans = {max_plans} → dừng.")
            break

    print(f"Hoàn tất: chọn {len(plan_cover)} plans, cover {len(covered)}/{len(universe)} instances")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plan_cover, f, indent=2)

    print(f"Đã lưu plan_cover → {out_path}")
    print("plan_cover =", plan_cover)


def main():
    parser = argparse.ArgumentParser(
        description="Build plan_cover cho q1_0 từ latency CSV."
    )
    parser.add_argument(
        "--lat",
        default="artifacts/exec_latencies_q1.csv",
        help="Đường dẫn file latency CSV (exec_latencies_q1.csv).",
    )
    parser.add_argument(
        "--out",
        default="data/plan_cover_q1.json",
        help="File output chứa list plan_id.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.1,
        help="Ngưỡng near-optimal (latency <= alpha * best_latency).",
    )
    parser.add_argument(
        "--max-plans",
        type=int,
        default=None,
        help="Giới hạn số lượng plan trong cover (None = không giới hạn).",
    )
    args = parser.parse_args()

    build_plan_cover(
        lat_path=args.lat,
        out_path=args.out,
        alpha=args.alpha,
        max_plans=args.max_plans,
    )


if __name__ == "__main__":
    main()
