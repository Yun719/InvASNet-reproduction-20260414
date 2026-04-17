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

try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_OK = True
except Exception:
    TENSORBOARD_OK = False

warnings.filterwarnings("ignore")
steps_per_epoch = 200 

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
      - tuple/list: (cover, secret)
      - dict: {"cover":..., "secret":...}
      - tensor: (B, C, L)  -> will split half/half (legacy)
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) != 2:
            raise ValueError("DataLoader 回傳 tuple/list 但長度不是 2，請改成 (cover, secret)")
        cover, secret = batch
    elif isinstance(batch, dict):
        cover, secret = batch["cover"], batch["secret"]
    else:
        # legacy fallback: one tensor, half is secret half is cover
        x = batch
        cover = x[x.shape[0] // 2 :]
        secret = x[: x.shape[0] // 2]

    cover = cover.to(device, non_blocking=True)
    secret = secret.to(device, non_blocking=True)
    return cover, secret


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
    channels_in = int(getattr(c, "channels_in", 1))

    lam_r = float(getattr(c, "lamda_reconstruction", 5.0))
    lam_g = float(getattr(c, "lamda_guide", 1.0))
    lam_l = float(getattr(c, "lamda_low_frequency", 1.0))

    # NOTE:
    # 2D HiNet 用 4*channels_in（因為 2D Haar: LL/LH/HL/HH）
    # 1D Haar 用 2*channels_in（因為 1D Haar: low/high）
    split_factor = 2

    print("=" * 80)
    print("Epoch    Loss        log10(lr)")
    print("=" * 80)

    try:
        for ep in range(epochs):
            i_epoch = trained_epoch + ep + 1
            net.train()

            loss_list = []
            g_list, r_list, l_list = [], [], []

            from tqdm import tqdm  # 引入進度條套件

            # 將原本的迴圈包裝進 tqdm 進度條中
            pbar = tqdm(enumerate(datasets.trainloader), total=steps_per_epoch, desc=f"Epoch {i_epoch}")
            for i_batch, batch in pbar:
                if i_batch >= steps_per_epoch:
                    break
                cover, secret = to_device_batch(batch, device)

                # 1) DWT
                cover_d = dwt(cover)     # (B, 2C, L/2)
                secret_d = dwt(secret)   # (B, 2C, L/2)
                check_finite("cover_d", cover_d)
                check_finite("secret_d", secret_d)

                # 2) concat
                x = torch.cat([cover_d, secret_d], dim=1)  # (B, 4C, L/2)

                # 3) forward (embed)
                y = net(x, rev=False)
                check_finite("y", y)

                y_steg = y.narrow(1, 0, split_factor * channels_in)  # (B, 2C, L/2)
                y_z = y.narrow(1, split_factor * channels_in, y.shape[1] - split_factor * channels_in)

                steg = iwt(y_steg)  # back to waveform (B, C, L)
                check_finite("steg", steg)

                # 4) backward (recover)
                z_rand = gauss_noise_like(y_z)
                y_rev_in = torch.cat([y_steg, y_z], dim=1)
                x_hat = net(y_rev_in, rev=True)
                check_finite("x_hat", x_hat)

                secret_hat_d = x_hat.narrow(
                    1, split_factor * channels_in, x_hat.shape[1] - split_factor * channels_in
                )
                secret_hat = iwt(secret_hat_d)
                check_finite("secret_hat", secret_hat)

                # 5) losses (照原 HiNet 的三個 loss 形式搬過來)
                g_loss = mse_loss_mean(steg, cover)                # steg 要像 cover
                r_loss = mse_loss_mean(secret_hat, secret)         # recover secret
                steg_low = y_steg.narrow(1, 0, channels_in)         # 1D low band
                cover_low = cover_d.narrow(1, 0, channels_in)
                l_loss = mse_loss_mean(steg_low, cover_low)

                total = lam_r * r_loss + lam_g * g_loss + lam_l * l_loss
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

                # 讓進度條旁邊即時顯示最新的誤差值
                pbar.set_postfix({"Loss": f"{total.item():.6f}"})
            # epoch stats
            epoch_loss = float(np.mean(loss_list)) if len(loss_list) else float("nan")
            lr_log10 = float(math.log10(optimizer.param_groups[0]["lr"]))
            print(f"{i_epoch:04d}    {epoch_loss:.6f}   {lr_log10:.4f}")

            # viz / tensorboard
            viz.show_loss([epoch_loss, lr_log10])
            if writer is not None:
                writer.add_scalars("Train", {"Loss": epoch_loss}, i_epoch)
                writer.add_scalars("TrainParts", {
                    "g_loss": float(np.mean(g_list)) if g_list else 0.0,
                    "r_loss": float(np.mean(r_list)) if r_list else 0.0,
                    "l_loss": float(np.mean(l_list)) if l_list else 0.0,
                }, i_epoch)

            # simple val (可先關掉省時間)
            if val_freq > 0 and (i_epoch % val_freq == 0):
                net.eval()
                with torch.no_grad():
                    vloss = []
                    for batch in datasets.testloader:
                        cover, secret = to_device_batch(batch, device)
                        cover_d = dwt(cover)
                        secret_d = dwt(secret)
                        x = torch.cat([cover_d, secret_d], dim=1)
                        y = net(x, rev=False)

                        y_steg = y.narrow(1, 0, split_factor * channels_in)
                        y_z = y.narrow(1, split_factor * channels_in, y.shape[1] - split_factor * channels_in)

                        steg = iwt(y_steg)
                        z_rand = gauss_noise_like(y_z)
                        x_hat = net(torch.cat([y_steg, z_rand], dim=1), rev=True)
                        secret_hat = iwt(
                            x_hat.narrow(1, split_factor * channels_in, x_hat.shape[1] - split_factor * channels_in)
                        )

                        g_loss = mse_loss_mean(steg, cover)
                        r_loss = mse_loss_mean(secret_hat, secret)
                        steg_low = y_steg.narrow(1, 0, channels_in)
                        cover_low = cover_d.narrow(1, 0, channels_in)
                        l_loss = mse_loss_mean(steg_low, cover_low)
                        total = lam_r * r_loss + lam_g * g_loss + lam_l * l_loss
                        vloss.append(float(total.item()))

                    val_loss = float(np.mean(vloss)) if vloss else float("nan")
                    print(f"[VAL] epoch {i_epoch:04d} loss={val_loss:.6f}")
                    if writer is not None:
                        writer.add_scalars("Val", {"Loss": val_loss}, i_epoch)

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
        viz.signal_stop()


if __name__ == "__main__":
    main()
