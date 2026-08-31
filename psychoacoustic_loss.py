"""
psychoacoustic_loss.py  ─  Differentiable Psychoacoustic Perceptual Loss
=========================================================================
實作論文中提出的心理聲學感知損失函數：

    L_psy = (1/N) * Σ_k  max(0, |FFT(stego)[k] − FFT(cover)[k]| − M[k])²

其中 M[k] 為第 k 個頻率 bin 的全域遮蔽門限（Global Masking Threshold），
由 cover 訊號的功率頻譜密度（PSD）可微分估計而來：

    M[k] = alpha × smoothed_cover_magnitude[k] + T_abs[k]

- smoothed_cover_magnitude：對 cover 的 log-magnitude 做平均池化（模擬 Bark
  域的頻率遮蔽擴散效應），再轉回線性域。
- T_abs[k]：人耳絕對聽閾（Absolute Threshold of Hearing），依 ISO 226 /
  Terhardt 公式計算，確保低於人耳感知下限的極微小差異不會被懲罰。
- alpha：遮蔽比例，控制門限相對 cover 頻譜的嚴格程度（越小 = 越嚴格）。

當 stego 與 cover 的頻譜差異低於遮蔽門限 M[k] 時，損失為 0（感知透明）；
超出門限的部分才產生懲罰（hinge loss），迫使模型將嵌入噪音限制在人耳不可
察覺的範圍內。

使用方式：
    from psychoacoustic_loss import PsychoacousticLoss
    criterion = PsychoacousticLoss(sample_rate=44100, n_fft=4096).to(device)
    loss = criterion(steg, cover)   # steg, cover: (B, C, L) 時域 tensor
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


class PsychoacousticLoss(nn.Module):
    """
    可微分心理聲學感知損失（Global Masking Threshold Hinge Loss）。

    Parameters
    ----------
    sample_rate  : int   – 音訊採樣率（Hz），預設 44100
    n_fft        : int   – 每個分析幀的 FFT 點數（推薦 4096）。
                           必須 <= segment_length。
    alpha        : float – 遮蔽比例（signal-to-mask ratio 近似）。
                           0.05 ≈ −26 dB SMR（非常嚴格）；
                           0.20 ≈ −14 dB SMR（較寬鬆）。
                           推薦初始值：0.10。
    spread_width : int   – Bark 域頻率擴散核寬度（必須為奇數），
                           越大 = 遮蔽範圍越廣。推薦：31。
    ref_spl_db   : float – 振幅 1.0 對應的聲壓級（dB SPL）。標準參考值 94 dB。
    """

    def __init__(
        self,
        sample_rate:  int   = 44100,
        n_fft:        int   = 4096,
        alpha:        float = 0.10,
        spread_width: int   = 31,
        ref_spl_db:   float = 94.0,
    ):
        super().__init__()

        if spread_width % 2 == 0:
            raise ValueError(f"spread_width 必須為奇數，目前為 {spread_width}")
        if n_fft <= 0 or (n_fft & (n_fft - 1)) != 0:
            warnings.warn(
                f"n_fft={n_fft} 不是 2 的次方，建議改為 2048/4096/8192 以最佳化 FFT 效率。",
                stacklevel=2,
            )

        self.sample_rate  = sample_rate
        self.n_fft        = n_fft
        self.alpha        = alpha
        self.spread_width = spread_width

        # ── 絕對聽閾（Absolute Threshold of Hearing, ATH） ─────────────────
        # rfft 產生 n_fft // 2 + 1 個頻率 bin
        n_bins   = n_fft // 2 + 1
        freqs_hz = torch.linspace(0.0, sample_rate / 2.0, n_bins)    # (F,) in Hz

        # ISO 226 / Terhardt 近似公式（單位：dB SPL）
        f_khz = (freqs_hz / 1000.0).clamp(min=0.1)    # 0.1 kHz = 100 Hz，避免低頻分母發散
        T_abs_db = (
            3.64  * f_khz.pow(-0.8)
            - 6.5 * torch.exp(-0.6 * (f_khz - 3.3).pow(2))
            + 1e-3 * f_khz.pow(4)
        )
        # 鉗制到合理聽覺範圍 dB SPL（0~80 dB SPL）
        # 超過 80 dB 的「幾乎不可能察覺」頻率，不應有過強的遮蔽能力
        T_abs_db = T_abs_db.clamp(min=0.0, max=80.0)

        # dB SPL → 線性振幅（ref_spl_db dB SPL 對應振幅 1.0）
        T_abs_linear = 10.0 ** ((T_abs_db - ref_spl_db) / 20.0)
        T_abs_linear = T_abs_linear.clamp(min=1e-9)

        # Hann 窗（減少頻譜洩漏）
        window = torch.hann_window(n_fft)

        # register_buffer：隨 .to(device) 自動搬移，且存入 state_dict
        self.register_buffer("T_abs",  T_abs_linear)   # (F,)
        self.register_buffer("window", window)          # (n_fft,)

    # ─────────────────────────────────────────────────────────────────────────
    def _masking_threshold(self, cover_frame: torch.Tensor) -> torch.Tensor:
        """
        從 cover 的單一分析幀估計可微分遮蔽門限 M。

        Args
            cover_frame : (N, n_fft)   N = B × C
        Returns
            M           : (N, F)       每個頻率 bin 的遮蔽門限（線性振幅）
        """
        fft_c = torch.fft.rfft(cover_frame * self.window, n=self.n_fft)
        mag_c = torch.abs(fft_c)                                    # (N, F)

        # ── 頻率擴散函數（Spreading Function）────────────────────────────────
        # 在 log-magnitude 域做均值池化，模擬 Bark 尺度上的遮蔽擴散：
        # 某頻率的強音訊號會使鄰近頻率也難以被人耳察覺。
        log_mag        = torch.log(mag_c + 1e-9)                    # (N, F)
        log_mag_smooth = F.avg_pool1d(
            log_mag.unsqueeze(1),                                   # (N, 1, F)
            kernel_size=self.spread_width,
            stride=1,
            padding=self.spread_width // 2,
        ).squeeze(1)                                                # (N, F)
        mag_smooth = torch.exp(log_mag_smooth)                     # 回到線性域

        # 全域遮蔽門限 = α × 擴散後 cover 頻譜 + 人耳絕對聽閾
        M = self.alpha * mag_smooth + (self.T_abs.unsqueeze(0) * self.n_fft)      # (N, F)
        return M

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, stego: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
        """
        計算 L_psy。

        Args
            stego : (B, C, L)  stego 音訊（時域，經 IWT 還原後）
            cover : (B, C, L)  cover 音訊（時域，網路輸入）
        Returns
            loss  : scalar Tensor（可反傳梯度）
        """
        B, C, L = stego.shape
        N = B * C

        # 展平 batch 與 channel → (N, L)
        stego_flat = stego.reshape(N, L)
        cover_flat = cover.reshape(N, L)

        n_fft    = self.n_fft
        # 將整段音訊切成多個不重疊的分析幀，確保覆蓋整段音訊
        # e.g. L=44160, n_fft=4096 → 10 幀（覆蓋 40960 個樣本，≈ 93%）
        n_frames = max(1, L // n_fft)

        frame_losses = []
        for i in range(n_frames):
            start = i * n_fft
            end   = start + n_fft
            if end > L:
                break   # 最後不足一幀的殘餘部分捨棄，避免 zero-padding 影響估計

            s_frame = stego_flat[:, start:end]   # (N, n_fft)
            c_frame = cover_flat[:, start:end]   # (N, n_fft)

            # ── FFT（取消 norm="forward"，避免數值除以 4096 後趨近於零）───────
            fft_s = torch.fft.rfft(s_frame * self.window, n=n_fft)
            fft_c = torch.fft.rfft(c_frame * self.window, n=n_fft)

            # 複數頻譜差的模（同時考慮振幅差與相位差）
            diff = torch.abs(fft_s - fft_c)                        # (N, F)

            # ── 遮蔽門限 ─────────────────────────────────────────────────────
            M = self._masking_threshold(c_frame)                    # (N, F)

            # ── Hinge Loss（採用 L1 超額，避免平方讓極小微量消失）────────────
            excess = torch.relu(diff - M)                           # (N, F)
            frame_losses.append((excess / n_fft).mean())

        # 對所有分析幀取平均
        if not frame_losses:
            return stego.sum() * 0.0   # 保持梯度圖連通，值為 0
        return torch.stack(frame_losses).mean()
