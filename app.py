import os
import torch
import torchaudio
import torchaudio.functional as F
import gradio as gr

try:
    from scipy.signal import butter, sosfilt
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    print("[警告] scipy 未安裝，Butterworth 濾波器將降級為 biquad")

try:
    import noisereduce as nr
    import numpy as np
    _HAS_NR = True
except ImportError:
    _HAS_NR = False
    print("[警告] noisereduce 未安裝，頻譜閘控將降級為 biquad。可執行: pip install noisereduce")

# ---- Whisper ASR 後端偵測 ----------------------------------------
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    _WHISPER_BACKEND = "faster-whisper"
    print("[ASR] 使用後端: faster-whisper")
except ImportError:
    FasterWhisperModel = None
    try:
        import whisper as _openai_whisper
        _WHISPER_BACKEND = "openai-whisper"
        print("[ASR] 使用後端: openai-whisper")
    except ImportError:
        _openai_whisper = None
        _WHISPER_BACKEND = None
        print("[警告] 未安裝任何 Whisper 套件，語音辨識功能不可用。"
              "可執行: pip install faster-whisper")
# -----------------------------------------------------------------

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
def hide_audio(cover_path, cover_vol, secret1_path, secret_vol, secret2_path=None):
    if not cover_path or not secret1_path: return None, "請上傳檔案！"
    try:
        cover   = load_and_preprocess(cover_path)
        # Host 音量微調倍率（預設 1.0 = 不額外調整）
        cover   = cover * cover_vol
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
        return output_path, f"✅ 隱寫成功！({rms_info}, Host倍率={cover_vol}, Secret微調={secret_vol}倍)"
    except Exception as e:
        return None, f"❌ 發生錯誤: {str(e)}"


@torch.no_grad()
def extract_audio(stego_path, extract_vol, filter_mode, stft_quantile):
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

            if filter_mode == "無濾波":
                # 不做任何濾波處理
                pass

            elif filter_mode == "Biquad 帶通（原始）":
                # 原始方案：biquad 一階 IIR，300–3400 Hz 電話頻帶
                audio = F.highpass_biquad(audio, target_sr, 300.0)
                audio = F.lowpass_biquad(audio,  target_sr, 3400.0)

            elif filter_mode == "Butterworth 帶通（scipy）":
                # 8 階 Butterworth，滾降更陡，頻率外雜訊壓制更徹底
                if _HAS_SCIPY:
                    a_np = audio.squeeze().cpu().float().numpy()
                    sos = butter(N=8, Wn=[300, min(8000, target_sr // 2 - 1)],
                                 btype='bandpass', fs=target_sr, output='sos')
                    a_np = sosfilt(sos, a_np)
                    audio = torch.tensor(a_np, dtype=audio.dtype, device=audio.device)
                    audio = audio.unsqueeze(0).unsqueeze(0)
                else:
                    print("[降級] scipy 未安裝，改用 biquad")
                    audio = F.highpass_biquad(audio, target_sr, 300.0)
                    audio = F.lowpass_biquad(audio,  target_sr, 3400.0)

            elif filter_mode == "STFT 頻譜減法（無依賴）":
                # 純 torch：對頻域低能量 bin 做軟式閘控抑制
                wav = audio.squeeze()          # (L,)
                n_fft = 1024
                stft = torch.stft(wav, n_fft=n_fft, hop_length=256,
                                  win_length=n_fft,
                                  window=torch.hann_window(n_fft, device=device),
                                  return_complex=True)   # (F, T)
                magnitude = stft.abs()
                # 用第 70 百分位作為門限：抑制能量最低的 70% bin，保留最強的 30%。
                # 原本 median×0.5 太寬鬆（門限低於中位數），幾乎沒有過濾效果。
                threshold = magnitude.flatten().quantile(float(stft_quantile))
                mask = torch.clamp((magnitude - threshold) / (threshold + 1e-8), 0.0, 1.0)
                stft_denoised = stft * mask
                wav_out = torch.istft(stft_denoised, n_fft=n_fft, hop_length=256,
                                      win_length=n_fft,
                                      window=torch.hann_window(n_fft, device=device),
                                      length=wav.shape[0])
                audio = wav_out.unsqueeze(0).unsqueeze(0)

            elif filter_mode == "Spectral Gating（noisereduce）":
                # 使用 noisereduce：估算雜訊底板並在頻域做自適應減法
                if _HAS_NR:
                    a_np = audio.squeeze().cpu().float().numpy()
                    a_np = nr.reduce_noise(y=a_np, sr=target_sr, stationary=False,
                                           prop_decrease=0.8)
                    audio = torch.tensor(a_np, dtype=audio.dtype, device=audio.device)
                    audio = audio.unsqueeze(0).unsqueeze(0)
                    print("CIallo~")
                else:
                    print("[降級] noisereduce 未安裝，改用 biquad。請執行: pip install noisereduce")
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
            msg = f"✅ 提取成功！兩段秘密音訊已還原 (濾波: {filter_mode})"
            return out1, out2, msg
        else:
            secret1_hat = secret_hat_all
            for _ in range(haar_levels):
                secret1_hat = iwt(secret1_hat)
            secret1_hat = postprocess(secret1_hat)
            out1 = "output_recovered_secret.wav"
            torchaudio.save(out1, secret1_hat.squeeze(0).cpu(), target_sr)
            msg = f"✅ 提取成功！(濾波模式: {filter_mode})"
            return out1, None, msg
    except Exception as e:
        return None, None, f"❌ 發生錯誤: {str(e)}"


# ==========================================
# 5. Whisper 語音辨識
# ==========================================
_whisper_cache: dict = {}  # {(backend, model_size, device_str): model_instance}

def transcribe_audio(audio_path, model_size: str, language: str) -> str:
    """
    對指定音訊檔進行語音辨識並回傳文字。
    audio_path : 音訊檔路徑（字串）
    model_size : tiny / base / small / medium / large
    language   : auto / zh / en / ja / ko …
    """
    if _WHISPER_BACKEND is None:
        return "❌ 未安裝 Whisper 套件，請執行: pip install faster-whisper"
    if not audio_path or not os.path.exists(str(audio_path)):
        return "⚠️ 尚無可辨識的音訊，請先執行提取（Extract）"

    lang = None if language == "auto" else language
    cache_key = (_WHISPER_BACKEND, model_size, str(device))

    try:
        if _WHISPER_BACKEND == "faster-whisper":
            if cache_key not in _whisper_cache:
                print(f"[ASR] 載入 faster-whisper 模型: {model_size}")
                compute = "float16" if torch.cuda.is_available() else "int8"
                _whisper_cache[cache_key] = FasterWhisperModel(
                    model_size,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    compute_type=compute,
                )
            model = _whisper_cache[cache_key]
            segments, info = model.transcribe(
                str(audio_path),
                language=lang,
                beam_size=5,
            )
            detected_lang = info.language
            text = "".join(seg.text for seg in segments).strip()
            return f"🌐 偵測語言: {detected_lang}\n\n📝 辨識結果:\n{text if text else '（無法辨識出有效文字）'}"

        else:  # openai-whisper
            if cache_key not in _whisper_cache:
                print(f"[ASR] 載入 openai-whisper 模型: {model_size}")
                _whisper_cache[cache_key] = _openai_whisper.load_model(
                    model_size,
                    device=str(device),
                )
            model = _whisper_cache[cache_key]
            result = model.transcribe(
                str(audio_path),
                language=lang,
            )
            detected_lang = result.get("language", "unknown")
            text = result.get("text", "").strip()
            return f"🌐 偵測語言: {detected_lang}\n\n📝 辨識結果:\n{text if text else '（無法辨識出有效文字）'}"

    except Exception as e:
        return f"❌ 辨識失敗: {str(e)}"


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
                    cover_vol  = gr.Slider(minimum=0.1, maximum=3.0, value=1.0, step=0.01,
                                           label="🎵 Host 音量微調倍率 (1.0=不額外調整)")
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
                inputs=[in_cover, cover_vol, in_secret, secret_vol, in_secret2],
                outputs=[out_stego, out_hide_msg]
            )

        with gr.TabItem("🔓 提取秘密 (Extract)"):
            with gr.Row():
                with gr.Column():
                    in_stego     = gr.Audio(label="Stego 音樂 (含秘密)", type="filepath")
                    extract_vol  = gr.Slider(minimum=1.0, maximum=20.0, value=5.0, step=1.0,
                                             label="📢 提取後放大倍率 (建議 5)")
                    _nr_note = "" if _HAS_NR else " ⚠️(需 pip install noisereduce)"
                    _sc_note = "" if _HAS_SCIPY else " ⚠️(需 pip install scipy)"
                    filter_mode  = gr.Dropdown(
                        choices=[
                            "無濾波",
                            "Biquad 帶通（原始）",
                            f"Butterworth 帶通（scipy）{_sc_note}",
                            "STFT 頻譜減法（無依賴）",
                            f"Spectral Gating（noisereduce）{_nr_note}",
                        ],
                        value="STFT 頻譜減法（無依賴）",
                        label="🎚️ 降噪濾波模式",
                        info="Butterworth > STFT > Biquad，noisereduce 效果最佳但需額外安裝"
                    )
                    stft_quantile = gr.Slider(
                        minimum=0.10, maximum=0.99, value=0.70, step=0.01,
                        label="🔢 STFT 門限百分位 (quantile)",
                        info="押制能量最低的 N% bin。大=更強力過濾，不這模式則此滾桿無效。",
                        visible=True   # 初始顯示，切換模式時動態隱藏
                    )
                    btn_extract  = gr.Button("開始提取 (Extract)", variant="primary")

                    gr.Markdown("---")
                    gr.Markdown("### 🗣️ Whisper 語音辨識設定")
                    _whisper_note = "" if _WHISPER_BACKEND else " ⚠️(需 pip install faster-whisper)"
                    whisper_model = gr.Dropdown(
                        choices=["tiny", "base", "small", "medium", "large"],
                        value="small",
                        label=f"Whisper 模型大小{_whisper_note}",
                        info="tiny/base=快, small=平衡(推薦), medium/large=最準但慢",
                        interactive=(_WHISPER_BACKEND is not None),
                    )
                    whisper_lang = gr.Dropdown(
                        choices=["auto", "zh", "en", "ja", "ko", "fr", "de", "es"],
                        value="auto",
                        label="辨識語言",
                        info="auto = Whisper 自動偵測",
                        interactive=(_WHISPER_BACKEND is not None),
                    )

                with gr.Column():
                    out_recovered  = gr.Audio(label="解碼出的 Secret 音樂 1", type="filepath")
                    btn_transcribe1 = gr.Button(
                        "🗣️ 辨識 Secret 1",
                        variant="secondary",
                        interactive=(_WHISPER_BACKEND is not None),
                    )
                    whisper_out1 = gr.Textbox(
                        label="📝 Secret 1 辨識結果",
                        interactive=False,
                        lines=4,
                        placeholder="請先提取音訊，再點擊辨識按鈕…",
                    )

                    out_recovered2 = gr.Audio(
                        label="解碼出的 Secret 音樂 2",
                        type="filepath",
                        visible=(num_secrets == 2),
                    )
                    btn_transcribe2 = gr.Button(
                        "🗣️ 辨識 Secret 2",
                        variant="secondary",
                        visible=(num_secrets == 2),
                        interactive=(_WHISPER_BACKEND is not None),
                    )
                    whisper_out2 = gr.Textbox(
                        label="📝 Secret 2 辨識結果",
                        interactive=False,
                        lines=4,
                        visible=(num_secrets == 2),
                        placeholder="請先提取音訊，再點擊辨識按鈕…",
                    )

                    out_extract_msg = gr.Textbox(label="系統訊息", interactive=False)

            # 切換濾波模式時，動態顯示/隱藏 STFT quantile slider
            filter_mode.change(
                fn=lambda m: gr.update(visible=(m == "STFT 頻譜減法（無依賴）")),
                inputs=[filter_mode],
                outputs=[stft_quantile],
            )
            btn_extract.click(
                fn=extract_audio,
                inputs=[in_stego, extract_vol, filter_mode, stft_quantile],
                outputs=[out_recovered, out_recovered2, out_extract_msg]
            )
            # Whisper 辨識按鈕綁定
            btn_transcribe1.click(
                fn=transcribe_audio,
                inputs=[out_recovered, whisper_model, whisper_lang],
                outputs=[whisper_out1],
            )
            btn_transcribe2.click(
                fn=transcribe_audio,
                inputs=[out_recovered2, whisper_model, whisper_lang],
                outputs=[whisper_out2],
            )

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)