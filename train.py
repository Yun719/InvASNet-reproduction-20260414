#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InvASNet (audio, 1D) training script
- expects datasets.trainloader / datasets.testloader to yield (cover, secret)
  cover:  (B, C, L)
  secret: (B, C, L)
- uses modules.dwt1d.DWT1D / IWT1D
- CPU-safe (no .cuda() hardcode)
"""
import os
print("[RUNNING FILE]", os.path.abspath(__file__))

import os
import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config as c
import datasets
import viz

from model import Model, init_model
from modules.dwt1d import DWT1D, IWT1D
from psychoacoustic_loss import PsychoacousticLoss   # 心理聲學感知損失
from tqdm import tqdm  # 引入進度條套件

try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_OK = True
except Exception:
    TENSORBOARD_OK = False

warnings.filterwarnings("ignore")
steps_per_epoch = 500

# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p: str):
    if p and (not os.path.exists(p)):
        os.makedirs(p, exist_ok=True)


def gauss_noise_like(x: torch.Tensor) -> torch.Tensor:
    # standard normal noise, same shape/device/dtype as x
    return torch.randn_like(x)


def mse_loss_mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return nn.MSELoss(reduction="mean")(a, b)


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {"Total": total_num, "Trainable": trainable_num}


def load_ckpt(net, optimizer, path: str, map_location):
    state = torch.load(path, map_location=map_location)
    net_state = {k: v for k, v in state.get("net", {}).items() if "tmp_var" not in k}
    net.load_state_dict(net_state, strict=False)
    if optimizer is not None and "opt" in state:
        try:
            optimizer.load_state_dict(state["opt"])
        except Exception:
            print("[WARN] optimizer state not loaded (ok).")


def to_device_batch(batch, device):
    """
    batch can be:
      - tuple/list of 2: (cover, secret)     → num_secrets=1
      - tuple/list of 3: (cover, s1, s2)     → num_secrets=2
      - dict: {"cover":..., "secret":...}
      - tensor: (B, C, L)  -> will split half/half (legacy)
    Returns: tuple of tensors on device
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) not in (2, 3):
            raise ValueError(f"DataLoader 回傳 tuple/list 但長度為 {len(batch)}，"
                             f"請確認 config.num_secrets 設定（目前只支援 1 或 2）")
        return tuple(t.to(device, non_blocking=True) for t in batch)
    elif isinstance(batch, dict):
        cover  = batch["cover"].to(device, non_blocking=True)
        secret = batch["secret"].to(device, non_blocking=True)
        return (cover, secret)
    else:
        # legacy fallback: one tensor, half is secret half is cover
        x = batch
        cover  = x[x.shape[0] // 2:].to(device, non_blocking=True)
        secret = x[:x.shape[0] // 2].to(device, non_blocking=True)
        return (cover, secret)


def check_finite(name, t: torch.Tensor):
    if not torch.isfinite(t).all():
        raise RuntimeError(f"[NaN/Inf] {name} 出現 NaN/Inf，請先停下來修正。")


# -----------------------------
# Main
# -----------------------------
def main():
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()
    print(f"[Device] {device}")

    # Ensure output dirs (用 config 裡的路徑；若你改成 Windows 路徑也 OK)
    model_dir = getattr(c, "MODEL_PATH", "./checkpoints/")
    ensure_dir(model_dir)

    # tensorboard
    writer = None
    if TENSORBOARD_OK:
        try:
            writer = SummaryWriter(comment="InvASNet", filename_suffix="audio1d")
        except Exception:
            writer = None

    # build model
    net = Model().to(device)
    init_model(net)

    # DataParallel only if multi-gpu
    if use_cuda and len(getattr(c, "device_ids", [])) > 1:
        net = torch.nn.DataParallel(net, device_ids=c.device_ids)

    para = get_parameter_number(net)
    print(para)

    params_trainable = list(filter(lambda p: p.requires_grad, net.parameters()))
    optimizer = optim.Adam(
        params_trainable,
        lr=getattr(c, "lr", 3e-5),
        betas=getattr(c, "betas", (0.5, 0.999)),
        eps=1e-6,
        weight_decay=getattr(c, "weight_decay", 1e-5),
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=getattr(c, "weight_step", 1000),
        gamma=getattr(c, "gamma", 0.5),
    )

    # DWT/IWT (1D Haar)
    dwt = DWT1D()
    iwt = IWT1D()

    # 心理聲學感知損失函數（在時域比較 stego 與 cover 的频譜感知差異）
    psy_criterion = PsychoacousticLoss(
        sample_rate  = int(getattr(c, "host_sr",        44100)),
        n_fft        = int(getattr(c, "psy_n_fft",      4096)),
        alpha        = float(getattr(c, "psy_alpha",    0.10)),
        spread_width = int(getattr(c, "psy_spread_width", 31)),
    ).to(device)
    print(f"[PsyLoss] n_fft={psy_criterion.n_fft}, alpha={psy_criterion.alpha}, "
          f"spread_width={psy_criterion.spread_width}")

    # resume?
    if getattr(c, "tain_next", False):
        ckpt_path = os.path.join(model_dir, getattr(c, "suffix", "model.pt"))
        if os.path.exists(ckpt_path):
            print(f"[Resume] loading {ckpt_path}")
            load_ckpt(net, optimizer, ckpt_path, map_location=device)
        else:
            print(f"[Resume] 找不到 {ckpt_path}，改為從頭訓練")

    # hyper
    epochs = int(getattr(c, "epochs", 1000))
    trained_epoch = int(getattr(c, "trained_epoch", 0))
    save_freq = int(getattr(c, "SAVE_freq", 50))
    val_freq = int(getattr(c, "val_freq", 50))
    channels_in          = int(getattr(c, "channels_in",         1))
    haar_levels          = int(getattr(c, "haar_levels",          1))
    num_secrets          = int(getattr(c, "num_secrets",          1))
    quantize_simulation  = bool(getattr(c, "quantize_simulation", False))
    if quantize_simulation:
        print("[Train] 量化模擬已啟用（Quantization-Aware Training）")
    else:
        print("[Train] 量化模擬未啟用")
    print(f"[Train] num_secrets={num_secrets}, haar_levels={haar_levels}")


    lam_r = float(getattr(c, "lamda_reconstruction", 5.0))
    lam_g = float(getattr(c, "lamda_guide", 1.0))
    lam_l = float(getattr(c, "lamda_low_frequency", 1.0))
    lam_psy = float(getattr(c, "lamda_psy", 0.0))   # 0.0 則完全關閉心理聲學損失
    print(f"[Train] lam_r={lam_r}, lam_g={lam_g}, lam_l={lam_l}, lam_psy={lam_psy}")

    # NOTE:
    # 1D Haar DWT 每疊一次：通道 ×2、長度 ÷2
    # haar_levels 次後：每邊有 channels_in * 2^haar_levels 個 channel
    # split_factor = 2^haar_levels（steg 側佔的 channel 數 / channels_in）
    split_factor = 2 ** haar_levels

    print(f"haar_levels：{haar_levels},split_factor:{split_factor}")

    print("=" * 80)
    print("Epoch    Loss        log10(lr)")
    print("=" * 80)

    try:
        for ep in range(epochs):
            i_epoch = trained_epoch + ep + 1
            net.train()

            loss_list = []
            g_list, r_list, l_list, psy_list = [], [], [], []



            # 將原本的迴圈包裝進 tqdm 進度條中
            pbar = tqdm(enumerate(datasets.trainloader), total=steps_per_epoch, desc=f"Epoch {i_epoch}")
            for i_batch, batch in pbar:
                if i_batch >= steps_per_epoch:
                    break
                batch_tensors = to_device_batch(batch, device)
                cover   = batch_tensors[0]
                secret1 = batch_tensors[1]
                secret2 = batch_tensors[2] if num_secrets == 2 else None


                # 1) DWT（haar_levels 次疊加）
                cover_d, secret1_d = cover, secret1
                for _ in range(haar_levels):
                    cover_d   = dwt(cover_d)
                    secret1_d = dwt(secret1_d)
                if num_secrets == 2:
                    secret2_d = secret2
                    for _ in range(haar_levels):
                        secret2_d = dwt(secret2_d)
                check_finite("cover_d",  cover_d)
                check_finite("secret1_d", secret1_d)

                # 2) concat
                if num_secrets == 2:
                    x = torch.cat([cover_d, secret1_d, secret2_d], dim=1)  # (B, 6C, L/2)
                else:
                    x = torch.cat([cover_d, secret1_d], dim=1)             # (B, 4C, L/2)

                # 3) forward (embed)
                y = net(x, rev=False)
                check_finite("y", y)

                y_steg = y.narrow(1, 0, split_factor * channels_in)  # (B, 2C, L/2)
                y_z = y.narrow(1, split_factor * channels_in, y.shape[1] - split_factor * channels_in)

                steg = y_steg
                for _ in range(haar_levels):
                    steg = iwt(steg)   # 逐層還原回波形 (B, C, L)
                check_finite("steg", steg)

                # ✅ 量化模擬（由 config.quantize_simulation 控制）
                # 模擬 16-bit WAV 儲存時的精度損失：
                #   1. 映射到 [-32768, 32767] 整數範圍
                #   2. 加入均勻量化噪聲 n ~ U(-0.5, 0.5)
                #   3. clamp 後正規化回 [-1, 1]
                #   4. 重新做 haar_levels 次 DWT 回頻域，再送進反向網路
                if quantize_simulation:
                    noise = torch.zeros_like(steg).uniform_(-0.5, 0.5)
                    steg_q = torch.clamp(32768.0 * steg + noise, -32768, 32767) / 32768.0
                    check_finite("steg_q", steg_q)
                    y_steg_q = steg_q
                    for _ in range(haar_levels):
                        y_steg_q = dwt(y_steg_q)   # 量化後重新 DWT 回頻域
                else:
                    steg_q   = steg      # 無量化，steg_q 只是個別名供 g_loss 使用
                    y_steg_q = y_steg    # 直接用原始頻域 tensor

                # 4) backward (recover)
                z_rand = gauss_noise_like(y_z)
                y_rev_in = torch.cat([y_steg_q, z_rand], dim=1)
                x_hat = net(y_rev_in, rev=True)
                check_finite("x_hat", x_hat)

                secret_hat_all = x_hat.narrow(
                    1, split_factor * channels_in, x_hat.shape[1] - split_factor * channels_in
                )
                if num_secrets == 2:
                    half = secret_hat_all.shape[1] // 2
                    secret1_hat = secret_hat_all.narrow(1, 0, half)
                    secret2_hat = secret_hat_all.narrow(1, half, half)
                    for _ in range(haar_levels):
                        secret1_hat = iwt(secret1_hat)
                        secret2_hat = iwt(secret2_hat)
                    check_finite("secret1_hat", secret1_hat)
                    check_finite("secret2_hat", secret2_hat)
                else:
                    secret1_hat = secret_hat_all
                    for _ in range(haar_levels):
                        secret1_hat = iwt(secret1_hat)
                    check_finite("secret_hat", secret1_hat)

                # 5) losses (照原 HiNet 的三個 loss 形式搞過來)
                # 量化啟用時 g_loss 用 steg_q（讓網路學會抗拗量化誤差）
                g_loss = mse_loss_mean(steg_q, cover)
                if num_secrets == 2:
                    r_loss = mse_loss_mean(secret1_hat, secret1) + mse_loss_mean(secret2_hat, secret2)
                else:
                    r_loss = mse_loss_mean(secret1_hat, secret1)
                steg_low = y_steg.narrow(1, 0, channels_in)         # 1D low band
                cover_low = cover_d.narrow(1, 0, channels_in)
                l_loss = mse_loss_mean(steg_low, cover_low)

                # 心理聲學感知損失（在時域比較 steg 與 cover，懲罰超出遮蔽門限的領域）
                # 使用未量化的 steg（直接來自網路），讓梯度反傳路徑更举正
                if lam_psy > 0.0:
                    psy_loss = psy_criterion(steg, cover)
                    check_finite("psy_loss", psy_loss)
                else:
                    psy_loss = torch.tensor(0.0, device=device)

                total = lam_r * r_loss + lam_g * g_loss + lam_l * l_loss + lam_psy * psy_loss
                check_finite("total_loss", total)

                optimizer.zero_grad(set_to_none=True)
                total.backward()
                # 避免爆掉
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

                loss_list.append(float(total.item()))
                g_list.append(float(g_loss.item()))
                r_list.append(float(r_loss.item()))
                l_list.append(float(l_loss.item()))
                psy_list.append(float(psy_loss.item()))

                # 讓進度條旁邊即時顯示最新的誤差值
                pbar.set_postfix({"Loss": f"{total.item():.6f}"})
            # epoch stats
            epoch_loss = float(np.mean(loss_list)) if len(loss_list) else float("nan")
            lr_log10 = float(math.log10(optimizer.param_groups[0]["lr"]))
            g_avg   = float(np.mean(g_list))   if g_list   else float("nan")
            r_avg   = float(np.mean(r_list))   if r_list   else float("nan")
            l_avg   = float(np.mean(l_list))   if l_list   else float("nan")
            psy_avg = float(np.mean(psy_list)) if psy_list else float("nan")
            print(
                f"Epoch {i_epoch:04d} | "
                f"Total={epoch_loss:.6f} | "
                f"r={r_avg:.6f}  g={g_avg:.6f}  l={l_avg:.6f}  psy={psy_avg:.3e} | "
                f"log10(lr)={lr_log10:.4f}"
            )

            # viz / tensorboard
            #viz.show_loss([epoch_loss, lr_log10])
            if writer is not None:
                writer.add_scalars("Train", {"Loss": epoch_loss}, i_epoch)
                writer.add_scalars("TrainParts", {
                    "g_loss":   float(np.mean(g_list))   if g_list   else 0.0,
                    "r_loss":   float(np.mean(r_list))   if r_list   else 0.0,
                    "l_loss":   float(np.mean(l_list))   if l_list   else 0.0,
                    "psy_loss": float(np.mean(psy_list)) if psy_list else 0.0,
                }, i_epoch)

            # simple val (可先關掉省時間)
            if val_freq > 0 and (i_epoch % val_freq == 0):
                net.eval()
                with torch.no_grad():
                    vloss = []

                    # 🌟 這裡加上 tqdm 進度條 🌟
                    val_pbar = tqdm(datasets.testloader, desc=f"Val Epoch {i_epoch}")

                    for batch in val_pbar:
                        val_tensors = to_device_batch(batch, device)
                        cover   = val_tensors[0]
                        secret1 = val_tensors[1]
                        secret2 = val_tensors[2] if num_secrets == 2 else None
                        cover_d, secret1_d = cover, secret1
                        for _ in range(haar_levels):
                            cover_d   = dwt(cover_d)
                            secret1_d = dwt(secret1_d)
                        if num_secrets == 2:
                            secret2_d = secret2
                            for _ in range(haar_levels):
                                secret2_d = dwt(secret2_d)
                        if num_secrets == 2:
                            x = torch.cat([cover_d, secret1_d, secret2_d], dim=1)
                        else:
                            x = torch.cat([cover_d, secret1_d], dim=1)
                        y = net(x, rev=False)

                        y_steg = y.narrow(1, 0, split_factor * channels_in)
                        y_z = y.narrow(1, split_factor * channels_in, y.shape[1] - split_factor * channels_in)

                        steg = y_steg
                        for _ in range(haar_levels):
                            steg = iwt(steg)
                        z_rand = gauss_noise_like(y_z)
                        x_hat = net(torch.cat([y_steg, z_rand], dim=1), rev=True)
                        secret_hat_all = x_hat.narrow(1, split_factor * channels_in, x_hat.shape[1] - split_factor * channels_in)
                        if num_secrets == 2:
                            half = secret_hat_all.shape[1] // 2
                            secret1_hat = secret_hat_all.narrow(1, 0, half)
                            secret2_hat = secret_hat_all.narrow(1, half, half)
                            for _ in range(haar_levels):
                                secret1_hat = iwt(secret1_hat)
                                secret2_hat = iwt(secret2_hat)
                        else:
                            secret1_hat = secret_hat_all
                            for _ in range(haar_levels):
                                secret1_hat = iwt(secret1_hat)

                        g_loss = mse_loss_mean(steg, cover)
                        if num_secrets == 2:
                            r_loss = mse_loss_mean(secret1_hat, secret1) + mse_loss_mean(secret2_hat, secret2)
                        else:
                            r_loss = mse_loss_mean(secret1_hat, secret1)
                        steg_low = y_steg.narrow(1, 0, channels_in)
                        cover_low = cover_d.narrow(1, 0, channels_in)
                        l_loss = mse_loss_mean(steg_low, cover_low)
                        total = lam_r * r_loss + lam_g * g_loss + lam_l * l_loss
                        vloss.append(float(total.item()))

                        # 🌟 讓進度條旁邊即時顯示考試的 Loss 🌟
                        val_pbar.set_postfix({"Loss": f"{total.item():.6f}"})

            # save
            if save_freq > 0 and (i_epoch % save_freq == 0):
                save_path = os.path.join(model_dir, f"model_checkpoint_{i_epoch:05d}.pt")
                torch.save({"opt": optimizer.state_dict(), "net": net.state_dict()}, save_path)

            scheduler.step()

        # final save
        final_path = os.path.join(model_dir, "model.pt")
        torch.save({"opt": optimizer.state_dict(), "net": net.state_dict()}, final_path)
        print(f"[DONE] saved to {final_path}")

    except Exception as e:
        # abort save
        if getattr(c, "checkpoint_on_error", True):
            abort_path = os.path.join(model_dir, "model_ABORT.pt")
            try:
                torch.save({"opt": optimizer.state_dict(), "net": net.state_dict()}, abort_path)
                print(f"[ABORT] saved to {abort_path}")
            except Exception:
                pass
        raise e

    finally:
        if writer is not None:
            writer.close()
        #viz.signal_stop()


if __name__ == "__main__":
    main()
