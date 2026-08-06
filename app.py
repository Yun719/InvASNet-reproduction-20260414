import os
import torch
import torchaudio
import torchaudio.functional as F
import gradio as gr

import config as c
from model import Model
from modules.dwt1d import DWT1D, IWT1D

# ==========================================
# 1. 系統與模型初始化
# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[Gradio] 使用裝置: {device}")

net = Model().to(device)
dwt = DWT1D().to(device)
iwt = IWT1D().to(device)

model_path = os.path.join(getattr(c, "MODEL_PATH", "./model/"), "model.pt")
if os.path.exists(model_path):
    state = torch.load(model_path, map_location=device)
    net_state = {k.replace('module.', ''): v for k, v in state.get("net", {}).items()}
    net.load_state_dict(net_state, strict=False)
    print(f"[Gradio] 成功載入模型權重: {model_path}")
else:
    print(f"[警告] 找不到模型檔 {model_path}，將使用隨機權重（僅供測試介面）")

net.eval()

channels_in         = int(getattr(c, "channels_in", 1))
haar_levels         = int(getattr(c, "haar_levels", 1))
quantize_simulation = bool(getattr(c, "quantize_simulation", False))
num_secrets         = int(getattr(c, "num_secrets", 1))
split_factor        = 2 ** haar_levels
target_sr           = getattr(c, "host_sr", 44100)
secret_target_rms   = float(getattr(c, "secret_target_rms", 0.0))   # 嵌入前 RMS 正規化目標
print(f"[App] 量化模擬: {'ON' if quantize_simulation else 'OFF'}, haar_levels: {haar_levels}, num_secrets: {num_secrets}")
print(f"[App] secret RMS 正規化: {secret_target_rms if secret_target_rms > 0 else '關閉'}")


# ==========================================
# 2. 輔助函式
# ==========================================
def _rms_normalize(wav: torch.Tensor, target_rms: float) -> torch.Tensor:
    """
    將音訊正規化到指定的 RMS 響度。
    wav        : (..., L) tensor
    target_rms : 目標 RMS（0.0 表示不處理）
    """
    if target_rms <= 0.0:
        return wav
    rms = wav.pow(2).mean().sqrt()
    if rms < 1e-8:
        return wav
    return torch.clamp(wav * (target_rms / rms), -1.0, 1.0)

def load_and_preprocess(audio_path, target_length=None):
    if audio_path is None: return None
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    if target_length is not None:
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        elif waveform.shape[1] < target_length:
            pad_len = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
    else:
        valid_len = (waveform.shape[1] // 16) * 16
        waveform = waveform[:, :valid_len]
    return waveform.unsqueeze(0).to(device)


# ==========================================
# 3. 核心功能
# ==========================================
@torch.no_grad()
def hide_audio(cover_path, secret1_path, secret_vol, secret2_path=None):
    if not cover_path or not secret1_path: return None, "請上傳檔案！"
    try:
        cover   = load_and_preprocess(cover_path)
        secret1 = load_and_preprocess(secret1_path, target_length=cover.shape[2])

        # ① 先做自動 RMS 正規化（訓練/推理一致）
        secret1 = _rms_normalize(secret1, secret_target_rms)
        # ② 再套用使用者的微調倍率（預設 1.0 = 不額外調整）
        secret1 = secret1 * secret_vol

        cover_d, secret1_d = cover, secret1
        for _ in range(haar_levels):
            cover_d   = dwt(cover_d)
            secret1_d = dwt(secret1_d)

        if num_secrets == 2:
            if not secret2_path:
                return None, "❌ num_secrets=2 但未上傳第二個秘密音訊！"
            secret2 = load_and_preprocess(secret2_path, target_length=cover.shape[2])
            secret2 = _rms_normalize(secret2, secret_target_rms)   # 自動正規化
            secret2 = secret2 * secret_vol                          # 微調倍率
            secret2_d = secret2
            for _ in range(haar_levels):
                secret2_d = dwt(secret2_d)
            x = torch.cat([cover_d, secret1_d, secret2_d], dim=1)
        else:
            x = torch.cat([cover_d, secret1_d], dim=1)

        y      = net(x, rev=False)
        y_steg = y.narrow(1, 0, split_factor * channels_in)
        steg_audio = y_steg
        for _ in range(haar_levels):
            steg_audio = iwt(steg_audio)
        # 量化模擬（由 config.quantize_simulation 控制）
        if quantize_simulation:
            steg_audio = torch.clamp(torch.round(32768.0 * steg_audio), -32768, 32767) / 32768.0
        else:
            steg_audio = torch.clamp(steg_audio, min=-1.0, max=1.0)

        output_path = "output_stego.wav"
        torchaudio.save(output_path, steg_audio.squeeze(0).cpu(), target_sr)
        rms_info = f"RMS正規化={secret_target_rms}" if secret_target_rms > 0 else "未正規化"
        return output_path, f"✅ 隱寫成功！({rms_info}, 微調={secret_vol}倍)"
    except Exception as e:
        return None, f"❌ 發生錯誤: {str(e)}"


@torch.no_grad()
def extract_audio(stego_path, extract_vol, apply_filter):
    if not stego_path: return None, None, "請上傳檔案！"
    try:
        steg = load_and_preprocess(stego_path)
        steg_d = steg
        for _ in range(haar_levels):
            steg_d = dwt(steg_d)

        # z_rand 必須與訓練時的 y_z 通道數相同
        # y_z 通道數 = split_factor * channels_in * num_secrets
        z_ch   = split_factor * channels_in * num_secrets
        z_rand = torch.randn(steg_d.shape[0], z_ch, steg_d.shape[2],
                             device=device, dtype=steg_d.dtype)

        y_rev_in = torch.cat([steg_d, z_rand], dim=1)
        x_hat    = net(y_rev_in, rev=True)
        secret_hat_all = x_hat.narrow(1, split_factor * channels_in,
                                       x_hat.shape[1] - split_factor * channels_in)

        def postprocess(audio):
            audio = audio * extract_vol
            if apply_filter:
                audio = F.highpass_biquad(audio, target_sr, 300.0)
                audio = F.lowpass_biquad(audio,  target_sr, 3400.0)
            return torch.clamp(audio, min=-1.0, max=1.0)

        if num_secrets == 2:
            half        = secret_hat_all.shape[1] // 2
            secret1_hat = secret_hat_all.narrow(1, 0, half)
            secret2_hat = secret_hat_all.narrow(1, half, half)
            for _ in range(haar_levels):
                secret1_hat = iwt(secret1_hat)
                secret2_hat = iwt(secret2_hat)
            secret1_hat = postprocess(secret1_hat)
            secret2_hat = postprocess(secret2_hat)
            out1 = "output_recovered_secret1.wav"
            out2 = "output_recovered_secret2.wav"
            torchaudio.save(out1, secret1_hat.squeeze(0).cpu(), target_sr)
            torchaudio.save(out2, secret2_hat.squeeze(0).cpu(), target_sr)
            msg = "✅ 提取成功！兩段秘密音訊已還原" + (" (已啟用人聲降噪濾波器)" if apply_filter else "")
            return out1, out2, msg
        else:
            secret1_hat = secret_hat_all
            for _ in range(haar_levels):
                secret1_hat = iwt(secret1_hat)
            secret1_hat = postprocess(secret1_hat)
            out1 = "output_recovered_secret.wav"
            torchaudio.save(out1, secret1_hat.squeeze(0).cpu(), target_sr)
            msg = "✅ 提取成功！" + (" (已啟用人聲增強濾波器)" if apply_filter else " (未啟用濾波)")
            return out1, None, msg
    except Exception as e:
        return None, None, f"❌ 發生錯誤: {str(e)}"


# ==========================================
# 4. Gradio 介面
# ==========================================
_mode_label = f"{'\u96d9\u79d8\u5bc6\u6a21\u5f0f' if num_secrets == 2 else '\u55ae\u79d8\u5bc6\u6a21\u5f0f'} (num_secrets={num_secrets})"

with gr.Blocks(title="InvASNet 隱寫測試平台") as app:
    gr.Markdown(f"# 🎵 InvASNet 音訊隱寫術測試平台\n### {_mode_label}")

    with gr.Tabs():
        with gr.TabItem("🔒 藏入音樂 (Hide)"):
            with gr.Row():
                with gr.Column():
                    in_cover   = gr.Audio(label="Host 音樂 (Cover)",                  type="filepath")
                    in_secret  = gr.Audio(label="Secret 音樂 1",                        type="filepath")
                    in_secret2 = gr.Audio(label="Secret 音樂 2 (num_secrets=2 時使用)",
                                          type="filepath", visible=(num_secrets == 2))
                    secret_vol = gr.Slider(minimum=0.1, maximum=3.0, value=1.0, step=0.01,
                                           label=f"🎧 秘密音量微調倍率 (已自動 RMS={secret_target_rms if secret_target_rms>0 else '未啟用'}, 1.0=不額外調整)")
                    btn_hide   = gr.Button("開始隱寫 (Hide)", variant="primary")
                with gr.Column():
                    out_stego    = gr.Audio(label="Stego 音樂 (含秘密)", type="filepath")
                    out_hide_msg = gr.Textbox(label="系統訊息", interactive=False)
            btn_hide.click(
                fn=hide_audio,
                inputs=[in_cover, in_secret, secret_vol, in_secret2],
                outputs=[out_stego, out_hide_msg]
            )

        with gr.TabItem("🔓 提取秘密 (Extract)"):
            with gr.Row():
                with gr.Column():
                    in_stego     = gr.Audio(label="Stego 音樂 (含秘密)", type="filepath")
                    extract_vol  = gr.Slider(minimum=1.0, maximum=20.0, value=5.0, step=1.0,
                                             label="📢 提取後放大倍率 (建議 5)")
                    apply_filter = gr.Checkbox(label="🎧 啟用人聲增強濾波器 (去除高低頻雜音)", value=True)
                    btn_extract  = gr.Button("開始提取 (Extract)", variant="primary")
                with gr.Column():
                    out_recovered  = gr.Audio(label="解碼出的 Secret 音樂 1", type="filepath")
                    out_recovered2 = gr.Audio(label="解碼出的 Secret 音樂 2",
                                              type="filepath", visible=(num_secrets == 2))
                    out_extract_msg = gr.Textbox(label="系統訊息", interactive=False)
            btn_extract.click(
                fn=extract_audio,
                inputs=[in_stego, extract_vol, apply_filter],
                outputs=[out_recovered, out_recovered2, out_extract_msg]
            )

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)