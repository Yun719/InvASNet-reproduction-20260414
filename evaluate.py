#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py - InvASNet 音訊隱寫品質評估工具（模型無關版）
==========================================================
本程式不需要載入任何模型。
直接對音訊檔案計算三項客觀指標：

  輸入（4 個音訊檔案）：
    cover      -- 原始載體音訊
    steg       -- 含密音訊（由任何版本的隱寫程式產生）
    secret     -- 原始秘密語音
    secret_hat -- 還原出的秘密語音

  輸出指標：
    SNR   -- 感知透明度，比較 cover vs steg
    PESQ  -- 語音清晰度，比較 secret vs secret_hat（ITU-T P.862）
    STOI  -- 語音可懂度，比較 secret vs secret_hat

執行方式：
  python evaluate.py
然後在瀏覽器開啟 http://127.0.0.1:7861
"""

import traceback

import numpy as np
import torch
import torchaudio
import gradio as gr

from pesq import pesq as _pesq
from pystoi import stoi as _stoi

EVAL_SR = 16000  # PESQ / STOI 固定需要 16 kHz

# ==========================================
# 1. 輔助函式
# ==========================================
def _load_mono_np(path, resample_to=None):
    """
    載入音訊，轉成單聲道，可選重採樣。
    回傳 (1D float32 ndarray, 採樣率)。
    """
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if resample_to is not None and sr != resample_to:
        waveform = torchaudio.functional.resample(waveform, sr, resample_to)
        sr = resample_to
    return waveform.squeeze().numpy().astype(np.float32), sr


def _align(a, b):
    """裁切兩個 array 到相同長度。"""
    n = min(len(a), len(b))
    return a[:n], b[:n]


def compute_snr(ref, deg):
    """Signal-to-Noise Ratio (dB)。越高代表含密音訊越接近原始載體。"""
    noise_power = np.sum((ref - deg) ** 2)
    if noise_power < 1e-12:
        return float("inf")
    return float(10.0 * np.log10(np.sum(ref ** 2) / noise_power))


def compute_pesq(ref16k, deg16k):
    """
    PESQ (ITU-T P.862)，寬頻模式，範圍 [-0.5, 4.5]。
    輸入必須是 16 kHz 的 numpy 1D float32 array。
    """
    try:
        n = min(len(ref16k), len(deg16k))
        return float(_pesq(EVAL_SR, ref16k[:n], deg16k[:n], "wb"))
    except Exception as e:
        print(f"[PESQ] 計算失敗: {e}")
        return None


def compute_stoi(ref16k, deg16k):
    """
    STOI，範圍 [0, 1]。
    輸入必須是 16 kHz 的 numpy 1D float32 array。
    """
    try:
        n = min(len(ref16k), len(deg16k))
        return float(_stoi(ref16k[:n], deg16k[:n], EVAL_SR, extended=False))
    except Exception as e:
        print(f"[STOI] 計算失敗: {e}")
        return None


def _grade(metric, value):
    thresholds = {
        "snr":  [(30, "優秀"), (20, "良好"), (10, "普通")],
        "pesq": [(3.5, "優秀"), (2.5, "良好"), (1.5, "普通")],
        "stoi": [(0.90, "優秀"), (0.70, "良好"), (0.50, "普通")],
    }
    icons = {"優秀": "🟢", "良好": "🟡", "普通": "🟠"}
    for thr, label in thresholds.get(metric, []):
        if value >= thr:
            return f"{icons[label]} {label}"
    return "🔴 差"


def _build_report(snr_val, pesq_val, stoi_val, cover_sr, error=None):
    if error:
        return f"### ❌ 評估失敗\n```\n{error}\n```"

    pesq_row = (
        f"| **PESQ** | `{pesq_val:.3f}` | {_grade('pesq', pesq_val)} | [-0.5 ~ 4.5]，越高語音越清晰 |"
        if pesq_val is not None else
        "| **PESQ** | `N/A` | ⚠️ 計算失敗 | 訊號可能太短或格式問題 |"
    )
    stoi_row = (
        f"| **STOI** | `{stoi_val:.4f}` | {_grade('stoi', stoi_val)} | [0 ~ 1]，越高語音越可懂 |"
        if stoi_val is not None else
        "| **STOI** | `N/A` | ⚠️ 計算失敗 | 訊號可能太短 |"
    )

    return f"""
---
### 🔇 感知透明度　（載體音訊 vs 藏密音訊，{cover_sr} Hz）

| 指標 | 數值 | 品質 | 說明 |
|:----:|:----:|:----:|------|
| **SNR** | `{snr_val:.2f} dB` | {_grade('snr', snr_val)} | 建議 > 20 dB，代表含密音訊與原始載體幾乎無差異 |

---
### 🗣️ 還原品質　（原始秘密 vs 還原秘密，計算前降採樣至 {EVAL_SR} Hz）

| 指標 | 數值 | 品質 | 說明 |
|:----:|:----:|:----:|------|
{pesq_row}
{stoi_row}

---
| 指標       | 🟢 優秀   | 🟡 良好     | 🟠 普通     | 🔴 差    |
| -------- | ------- | --------- | --------- | ------- |
| **SNR**  | > 30 dB | 20–30 dB  | 10–20 dB  | < 10 dB |
| **PESQ** | > 3.5   | 2.5–3.5   | 1.5–2.5   | < 1.5   |
| **STOI** | > 0.90  | 0.70–0.90 | 0.50–0.70 | < 0.50  |

"""


# ==========================================
# 2. 核心評估流程
# ==========================================
def run_evaluate(cover_path, steg_path, secret_path, secret_hat_path):
    """
    純音訊評估，不需要模型。
    1 次 Haar、3 次 Haar 等任何版本的輸出皆可使用。
    """
    missing = [name for name, p in [
        ("Cover",      cover_path),
        ("Steg",       steg_path),
        ("Secret",     secret_path),
        ("Secret_hat", secret_hat_path),
    ] if not p]
    if missing:
        return f"### ⚠️ 請上傳以下音訊：{'、'.join(missing)}"

    try:
        # SNR：以 Cover 的原始採樣率為基準，Steg 自動對齊
        cover_np, cover_sr = _load_mono_np(cover_path)
        steg_np,  _        = _load_mono_np(steg_path, resample_to=cover_sr)
        cover_a, steg_a    = _align(cover_np, steg_np)
        snr_val = compute_snr(cover_a, steg_a)

        # PESQ / STOI：降採樣到 16 kHz
        secret_16k,     _ = _load_mono_np(secret_path,     resample_to=EVAL_SR)
        secret_hat_16k, _ = _load_mono_np(secret_hat_path, resample_to=EVAL_SR)
        pesq_val = compute_pesq(secret_16k, secret_hat_16k)
        stoi_val = compute_stoi(secret_16k, secret_hat_16k)

        return _build_report(snr_val, pesq_val, stoi_val, cover_sr)

    except Exception:
        err = traceback.format_exc()
        print(err)
        return _build_report(0, None, None, 0, error=err)


# ==========================================
# 3. Gradio 介面
# ==========================================
with gr.Blocks(
    title="InvASNet 評估平台",
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.slate,
    ),
    css=".gr-button-primary { background: linear-gradient(135deg,#6366f1,#8b5cf6) !important; } footer { display: none !important; }",
) as demo:

    gr.Markdown(
        """
        # 🎵 InvASNet 音訊隱寫品質評估平台
        上傳 **4 個音訊檔案**，計算 **SNR / PESQ / STOI** 三項客觀指標。

        > 📌 本工具與模型架構完全無關，1 次 Haar、3 次 Haar 等任何版本輸出皆可評估。
        """
    )

    gr.Markdown("### 📥 輸入音訊（4 個）")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### 感知透明度評估用")
            in_cover = gr.Audio(
                label="🎼 Cover（原始載體音訊）",
                type="filepath",
            )
            in_steg = gr.Audio(
                label="🎭 Steg（含密音訊，由隱寫程式產生）",
                type="filepath",
            )
        with gr.Column(scale=1):
            gr.Markdown("#### 還原品質評估用")
            in_secret = gr.Audio(
                label="🔒 Secret（原始秘密語音）",
                type="filepath",
            )
            in_secret_hat = gr.Audio(
                label="🔓 Secret_hat（還原出的秘密語音）",
                type="filepath",
            )

    btn_eval = gr.Button("🚀 開始評估", variant="primary", size="lg")
    gr.Markdown("---")
    gr.Markdown("### 📊 評估指標")
    out_report = gr.Markdown(
        value="尚未評估，請上傳 4 個音訊後點擊開始評估。"
    )

    with gr.Accordion("📖 指標說明（展開查看）", open=False):
        gr.Markdown(
            f"""
            | 指標 | 全名 | 評估對象 | 範圍 | 建議目標 |
            |:----:|------|:--------:|:----:|:--------:|
            | **SNR** | Signal-to-Noise Ratio | 載體音訊 vs 藏密音訊 | dB（越高越好）| > 20 dB |
            | **PESQ** | Perceptual Evaluation of Speech Quality（ITU-T P.862）| 原始秘密 vs 還原秘密 | -0.5 ~ 4.5 | > 2.5 |
            | **STOI** | Short-Time Objective Intelligibility | 原始秘密 vs 還原秘密 | 0 ~ 1 | > 0.70 |

            #### 採樣率處理
            - **SNR** 以 Cover 的原始採樣率為基準，Steg 若不同會自動對齊。
            - **PESQ / STOI** 強制要求 **{EVAL_SR} Hz** 輸入，
              程式自動將 Secret 和 Secret_hat 重採樣後再計算。
            - **ODG**（未實作）：ITU-R BS.1387 目前無穩定 Python 套件，暫以 SNR 替代。
            """
        )

    btn_eval.click(
        fn=run_evaluate,
        inputs=[in_cover, in_steg, in_secret, in_secret_hat],
        outputs=[out_report],
    )


# ==========================================
# 5. 啟動
# ==========================================
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
