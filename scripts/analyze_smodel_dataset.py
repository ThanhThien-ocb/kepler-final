#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_smodel_dataset.py
-------------------------
Phân tích full dataset kết quả so sánh model vs default.

Input:  *_detail.csv
  - param_key
  - default_latency
  - optimal_latency
  - optimal_plan_id
  - pred_plan_id
  - model_latency
  - speedup_model
  - speedup_opt
  - selacc_flag

Output:
  - In thống kê ra màn hình
  - Ghi summary.txt
  - Vẽ:
      + histogram_delta.png  (delta = model_latency / optimal_latency)
      + histogram_speedup_model.png
      + scatter_default_vs_model.png

Cách chạy:
  python3 analyze_smodel_dataset.py \
    --csv artifacts/q1_0_detail.csv \
    --out-dir artifacts/analysis_q1_0
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def analyze(csv_path: str, out_dir: str) -> None:
    # ------------------------------------------------------
    # 1) Đọc dữ liệu
    # ------------------------------------------------------
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file CSV: {csv_path}")

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    required_cols = [
        "param_key",
        "default_latency",
        "optimal_latency",
        "optimal_plan_id",
        "pred_plan_id",
        "model_latency",
        "speedup_model",
        "speedup_opt",
        "selacc_flag",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Thiếu cột '{c}' trong file CSV.")

    n = len(df)
    print(f"✅ Đã load {n} dòng từ: {csv_path}")

    # ------------------------------------------------------
    # 2) Tính thêm các cột hữu ích
    # ------------------------------------------------------
    # Độ lệch so với optimal: delta = model_latency / optimal_latency
    df["delta"] = df["model_latency"] / df["optimal_latency"]

    # Regression: model chậm hơn default (speedup_model < 1)
    df["is_regression"] = df["speedup_model"] < 1.0

    # Model chọn đúng optimal (pred_plan_id == optimal_plan_id)
    df["is_exact_optimal"] = df["pred_plan_id"] == df["optimal_plan_id"]

    # ------------------------------------------------------
    # 3) Thống kê tổng quan
    # ------------------------------------------------------
    selacc = df["selacc_flag"].mean()  # tỉ lệ near-opt
    avg_speedup_model = df["speedup_model"].mean()
    avg_speedup_opt = df["speedup_opt"].mean()

    regression_rate = df["is_regression"].mean()
    exact_optimal_rate = df["is_exact_optimal"].mean()

    delta_mean = df["delta"].mean()
    delta_median = df["delta"].median()
    delta_max = df["delta"].max()
    delta_min = df["delta"].min()
    delta_q90 = df["delta"].quantile(0.9)
    delta_q95 = df["delta"].quantile(0.95)

    # ------------------------------------------------------
    # 4) In thống kê ra màn hình
    # ------------------------------------------------------
    print("\n=== 📊 THỐNG KÊ TỔNG QUAN ===")
    print(f"Số mẫu                         : {n}")
    print(f"SelAcc (tỉ lệ near-opt)        : {selacc:.4f}")
    print(f"Speedup(model) trung bình      : {avg_speedup_model:.4f} x")
    print(f"Speedup(optimal) trung bình    : {avg_speedup_opt:.4f} x")
    print(f"Tỉ lệ regression (<1.0x)       : {regression_rate*100:.2f} %")
    print(f"Tỉ lệ chọn đúng optimal plan   : {exact_optimal_rate*100:.2f} %")

    print("\n=== Δ = model_latency / optimal_latency ===")
    print(f"Mean(Δ)      : {delta_mean:.4f}")
    print(f"Median(Δ)    : {delta_median:.4f}")
    print(f"Min(Δ)       : {delta_min:.4f}")
    print(f"Max(Δ)       : {delta_max:.4f}")
    print(f"90th pct(Δ)  : {delta_q90:.4f}")
    print(f"95th pct(Δ)  : {delta_q95:.4f}")

    # Top 10 trường hợp tệ nhất (delta lớn)
    worst = df.sort_values("delta", ascending=False).head(10)

    # ------------------------------------------------------
    # 5) Ghi summary ra file
    # ------------------------------------------------------
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== SUMMARY MODEL vs DEFAULT ===\n")
        f.write(f"CSV nguồn             : {csv_path}\n")
        f.write(f"Số mẫu                : {n}\n")
        f.write(f"SelAcc                : {selacc:.6f}\n")
        f.write(f"Speedup(model)_mean   : {avg_speedup_model:.6f}\n")
        f.write(f"Speedup(optimal)_mean : {avg_speedup_opt:.6f}\n")
        f.write(f"Regression_rate       : {regression_rate:.6f}\n")
        f.write(f"Exact_optimal_rate    : {exact_optimal_rate:.6f}\n")
        f.write("\n--- Delta stats (model_latency / optimal_latency) ---\n")
        f.write(f"Mean(Δ)               : {delta_mean:.6f}\n")
        f.write(f"Median(Δ)             : {delta_median:.6f}\n")
        f.write(f"Min(Δ)                : {delta_min:.6f}\n")
        f.write(f"Max(Δ)                : {delta_max:.6f}\n")
        f.write(f"90th pct(Δ)           : {delta_q90:.6f}\n")
        f.write(f"95th pct(Δ)           : {delta_q95:.6f}\n")
        f.write("\n--- Worst 10 cases by Δ ---\n")
        for _, row in worst.iterrows():
            f.write(
                f"{row['param_key']}: Δ={row['delta']:.4f}, "
                f"default={row['default_latency']:.4f}, "
                f"optimal={row['optimal_latency']:.4f}, "
                f"model={row['model_latency']:.4f}, "
                f"speedup_model={row['speedup_model']:.4f}\n"
            )

    print(f"\n💾 Đã ghi summary vào: {summary_path}")

    # ------------------------------------------------------
    # 6) Vẽ histogram Δ
    # ------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(df["delta"], bins=40)
    plt.axvline(1.1, linestyle="--", label="Ngưỡng near-opt (Δ = 1.1)")
    plt.xlabel("Δ = model_latency / optimal_latency")
    plt.ylabel("Số lượng mẫu")
    plt.title("Histogram độ lệch giữa Model và Optimal (Δ)")
    plt.legend()

    hist_delta_path = os.path.join(out_dir, "histogram_delta.png")
    plt.tight_layout()
    plt.savefig(hist_delta_path, dpi=150)
    plt.close()
    print(f"📈 Đã lưu histogram Δ tại: {hist_delta_path}")

    # ------------------------------------------------------
    # 7) Vẽ histogram Speedup(model)
    # ------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(df["speedup_model"], bins=40)
    plt.axvline(1.0, linestyle="--", label="Mốc default (1.0x)")
    plt.xlabel("Speedup(model)")
    plt.ylabel("Số lượng mẫu")
    plt.title("Histogram Speedup(model) so với Default")
    plt.legend()

    hist_speedup_path = os.path.join(out_dir, "histogram_speedup_model.png")
    plt.tight_layout()
    plt.savefig(hist_speedup_path, dpi=150)
    plt.close()
    print(f"📈 Đã lưu histogram speedup tại: {hist_speedup_path}")

    # ------------------------------------------------------
    # 8) Scatter: default_latency vs model_latency
    # ------------------------------------------------------
    plt.figure(figsize=(6, 6))
    plt.scatter(df["default_latency"], df["model_latency"], s=5)
    max_lat = max(df["default_latency"].max(), df["model_latency"].max())
    plt.plot([0, max_lat], [0, max_lat], linestyle="--", label="Đường y=x")
    plt.xlabel("Default latency (ms)")
    plt.ylabel("Model latency (ms)")
    plt.title("Scatter: Default vs Model latency")
    plt.legend()

    scatter_path = os.path.join(out_dir, "scatter_default_vs_model.png")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"📈 Đã lưu scatter tại: {scatter_path}")

    print("\n✅ Phân tích hoàn tất.")


def main():
    parser = argparse.ArgumentParser(
        description="Phân tích full dataset kết quả model vs default."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Đường dẫn tới file *_detail.csv (output của 02_compare_default_and_model.py).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Thư mục để lưu summary.txt và các hình.",
    )
    args = parser.parse_args()
    analyze(args.csv, args.out_dir)


if __name__ == "__main__":
    main()
