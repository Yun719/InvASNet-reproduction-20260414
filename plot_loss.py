#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_loss.py  ─  InvASNet 訓練 Loss 視覺化工具
================================================
用法：
    python plot_loss.py                     # 使用預設路徑 ./loss_log.csv
    python plot_loss.py --csv ./loss_log.csv
    python plot_loss.py --smooth 10         # 移動平均窗口大小（預設 5）
    python plot_loss.py --out loss_chart.png  # 指定輸出圖片名稱

功能：
    - 讀取 train.py 寫入的 loss_log.csv
    - 支援中途中斷、隔天繼續的訓練記錄（epoch 連續追加，不重複）
    - 繪製 Total / g_loss / r_loss / l_loss / psy_loss / lr 六個子圖
    - 自動套用移動平均平滑曲線（原始數據半透明顯示在底層）
    - 圖表存檔到 PNG，同時彈出視窗預覽
"""

import argparse
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 無 GUI 環境也能存檔；若想同時彈視窗改成 "TkAgg"
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ─────────────────────────────────────────────
# 讀取 CSV
# ─────────────────────────────────────────────
def load_csv(csv_path: str) -> dict:
    """讀取 loss_log.csv，回傳 {欄位名: list} dict。"""
    if not os.path.exists(csv_path):
        print(f"[Error] 找不到 CSV 檔案：{csv_path}")
        sys.exit(1)

    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("[Error] CSV 檔案是空的或只有標題列")
        sys.exit(1)

    # 轉成 {col: [float, ...]}，跳過無法解析的列
    cols = rows[0].keys()
    data = {c: [] for c in cols}
    for row in rows:
        try:
            for c in cols:
                data[c].append(float(row[c]) if row[c] not in ("", "nan", "NaN") else float("nan"))
        except ValueError:
            continue  # 跳過格式異常的列

    print(f"[Info] 載入 {len(data['epoch'])} 個 epoch 記錄，來自 {csv_path}")
    return data


# ─────────────────────────────────────────────
# 移動平均
# ─────────────────────────────────────────────
def moving_avg(arr: list, w: int) -> np.ndarray:
    """對 arr 做寬度為 w 的中心移動平均（兩端用邊界值填補）。"""
    a = np.array(arr, dtype=float)
    if w <= 1 or len(a) < w:
        return a
    kernel = np.ones(w) / w
    # 使用 'same' 模式，兩端以最近有效值補齊
    pad = w // 2
    a_pad = np.concatenate([np.full(pad, a[~np.isnan(a)][0] if not np.all(np.isnan(a)) else 0),
                             a,
                             np.full(pad, a[~np.isnan(a)][-1] if not np.all(np.isnan(a)) else 0)])
    smoothed = np.convolve(a_pad, kernel, mode="valid")
    return smoothed[:len(a)]


# ─────────────────────────────────────────────
# 繪圖
# ─────────────────────────────────────────────
SUBPLOT_CFG = [
    # (欄位名,          子圖標題,               顏色,      y軸標籤)
    ("total_loss",   "Total Loss",           "#4C72B0", "Loss"),
    ("val_loss",     "Val Loss",             "#E377C2", "Loss"),
    ("g_loss",       "Guide Loss (g)",       "#DD8452", "Loss"),
    ("r_loss",       "Recon Loss (r)",       "#55A868", "Loss"),
    ("l_loss",       "LowFreq Loss (l)",     "#C44E52", "Loss"),
    ("psy_loss",     "Psychoacoustic Loss",  "#8172B2", "Loss"),
    ("lr",           "Learning Rate",        "#CCB974", "LR"),
]


def plot(data: dict, smooth_w: int, out_path: str):
    epochs = np.array(data["epoch"])

    # 決定有哪些欄位實際存在
    available = [(col, title, color, ylabel)
                 for col, title, color, ylabel in SUBPLOT_CFG
                 if col in data and not all(np.isnan(data[col]))]

    n = len(available)
    cols_per_row = 3
    rows = (n + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(rows, cols_per_row,
                             figsize=(6 * cols_per_row, 4 * rows),
                             squeeze=False)
    fig.suptitle("InvASNet Training Loss", fontsize=16, fontweight="bold", y=1.01)

    for i, (col, title, color, ylabel) in enumerate(available):
        ax = axes[i // cols_per_row][i % cols_per_row]
        raw = np.array(data[col])
        sm  = moving_avg(data[col], smooth_w)

        # 原始曲線（半透明）
        ax.plot(epochs, raw, color=color, alpha=0.25, linewidth=0.8, label="raw")
        # 平滑曲線
        ax.plot(epochs, sm,  color=color, alpha=0.95, linewidth=1.8,
                label=f"smoothed (w={smooth_w})")

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.4)

        # y 軸用科學記號（loss 值通常很小）
        if ylabel == "Loss":
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))

        # lr 用 log 軸
        if col == "lr":
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())

    # 隱藏多餘的空子圖
    for j in range(len(available), rows * cols_per_row):
        axes[j // cols_per_row][j % cols_per_row].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[Info] 圖表已存至：{os.path.abspath(out_path)}")

    # 若在有 GUI 的環境，嘗試彈視窗
    try:
        matplotlib.use("TkAgg")
        plt.show()
    except Exception:
        pass


# ─────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="InvASNet Loss 視覺化")
    p.add_argument("--csv",    default="./loss_log.csv", help="CSV 路徑（預設 ./loss_log.csv）")
    p.add_argument("--smooth", type=int, default=5,      help="移動平均窗口大小（預設 5）")
    p.add_argument("--out",    default="loss_chart.png", help="輸出圖片路徑（預設 loss_chart.png）")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = load_csv(args.csv)
    plot(data, smooth_w=args.smooth, out_path=args.out)
