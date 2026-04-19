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

channels_in = int(getattr(c, "channels_in", 1))
split_factor = 2
target_sr = getattr(c, "host_sr", 44100)


# ==========================================
# 2. 輔助函式
# ==========================================
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
def hide_audio(cover_path, secret_path, secret_vol):
    if not cover_path or not secret_path: return None, "請上傳檔案！"
    try:
        cover = load_and_preprocess(cover_path)
        secret = load_and_preprocess(secret_path, target_length=cover.shape[2])
        secret = secret * secret_vol

        cover_d = dwt(cover)
        secret_d = dwt(secret)
        x = torch.cat([cover_d, secret_d], dim=1)
        y = net(x, rev=False)
        y_steg = y.narrow(1, 0, split_factor * channels_in)
        steg_audio = iwt(y_steg)
        steg_audio = torch.clamp(steg_audio, min=-1.0, max=1.0)

        output_path = "output_stego.wav"
        torchaudio.save(output_path, steg_audio.squeeze(0).cpu(), target_sr)
        return output_path, f"✅ 隱寫成功！(音量縮小至 {secret_vol} 倍)"
    except Exception as e:
        return None, f"❌ 發生錯誤: {str(e)}"


@torch.no_grad()
def extract_audio(stego_path, extract_vol, apply_filter):
    if not stego_path: return None, "請上傳檔案！"
    try:
        steg = load_and_preprocess(stego_path)
        steg_d = dwt(steg)
        z_rand = torch.randn_like(steg_d)
        y_rev_in = torch.cat([steg_d, z_rand], dim=1)
        x_hat = net(y_rev_in, rev=True)
        secret_hat_d = x_hat.narrow(1, split_factor * channels_in, x_hat.shape[1] - split_factor * channels_in)
        secret_hat_audio = iwt(secret_hat_d)

        # 放大音量
        secret_hat_audio = secret_hat_audio * extract_vol

        # 🌟 【魔法降噪】啟用濾波器
        if apply_filter:
            # 切除 300Hz 以下低頻雜音
            secret_hat_audio = F.highpass_biquad(secret_hat_audio, target_sr, 300.0)
            # 切除 3400Hz 以上高頻嘶嘶聲
            secret_hat_audio = F.lowpass_biquad(secret_hat_audio, target_sr, 3400.0)

        secret_hat_audio = torch.clamp(secret_hat_audio, min=-1.0, max=1.0)
        output_path = "output_recovered_secret.wav"
        torchaudio.save(output_path, secret_hat_audio.squeeze(0).cpu(), target_sr)

        msg = "✅ 提取成功！" + (" (已啟用人聲降噪濾波)" if apply_filter else " (未啟用濾波)")
        return output_path, msg
    except Exception as e:
        return None, f"❌ 發生錯誤: {str(e)}"


# ==========================================
# 4. Gradio 介面
# ==========================================
with gr.Blocks(title="InvASNet 隱寫測試平台") as app:
    gr.Markdown("# 🎵 InvASNet 音訊隱寫術測試平台")

    with gr.Tabs():
        with gr.TabItem("🔒 藏入音樂 (Hide)"):
            with gr.Row():
                with gr.Column():
                    in_cover = gr.Audio(label="Host 音樂 (Cover)", type="filepath")
                    in_secret = gr.Audio(label="Secret 音樂 (Secret)", type="filepath")
                    secret_vol = gr.Slider(minimum=0.01, maximum=1.0, value=0.2, step=0.05,
                                           label="🗜️ 秘密音量縮放 (建議 0.2)")
                    btn_hide = gr.Button("開始隱寫 (Hide)", variant="primary")
                with gr.Column():
                    out_stego = gr.Audio(label="Stego 音樂 (含秘密)", type="filepath")
                    out_hide_msg = gr.Textbox(label="系統訊息", interactive=False)
            btn_hide.click(fn=hide_audio, inputs=[in_cover, in_secret, secret_vol], outputs=[out_stego, out_hide_msg])

        with gr.TabItem("🔓 提取秘密 (Extract)"):
            with gr.Row():
                with gr.Column():
                    in_stego = gr.Audio(label="Stego 音樂 (含秘密)", type="filepath")
                    extract_vol = gr.Slider(minimum=1.0, maximum=20.0, value=5.0, step=1.0,
                                            label="📢 提取後放大倍率 (建議 5)")
                    # 🌟 新增濾波器開關
                    apply_filter = gr.Checkbox(label="🎧 啟用人聲增強濾波器 (去除高低頻雜音)", value=True)
                    btn_extract = gr.Button("開始提取 (Extract)", variant="primary")
                with gr.Column():
                    out_recovered = gr.Audio(label="解碼出的 Secret 音樂", type="filepath")
                    out_extract_msg = gr.Textbox(label="系統訊息", interactive=False)
            btn_extract.click(fn=extract_audio, inputs=[in_stego, extract_vol, apply_filter],
                              outputs=[out_recovered, out_extract_msg])

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)