import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import config as c
import torchaudio


#print("[DEBUG] PWD =", os.getcwd())
#print("[DEBUG] c.TRAIN_HOST_PATH =", c.TRAIN_HOST_PATH, "exists=", os.path.exists(c.TRAIN_HOST_PATH))
#print("[DEBUG] c.TRAIN_SECRET_PATH =", c.TRAIN_SECRET_PATH, "exists=", os.path.exists(c.TRAIN_SECRET_PATH))
#print("[DEBUG] host files =", os.listdir(c.TRAIN_HOST_PATH) if os.path.exists(c.TRAIN_HOST_PATH) else None)
#print("[DEBUG] secret files =", os.listdir(c.TRAIN_SECRET_PATH) if os.path.exists(c.TRAIN_SECRET_PATH) else None)

# datasets.py 唯的一任務，就是去硬碟裡把你存放的 .wav 音檔挖出來，把它們切得整整齊齊，然後打包送進顯示卡裡給大廚烹飪。

def _resolve_path(root, item):
    # item 可能是 "host_000.wav" 或 "./data/val/host/host.wav" 或 "/content/InvASNet/..."
    item = str(item)

    # 1) 絕對路徑：直接用
    if os.path.isabs(item) and os.path.exists(item):
        return item

    # 2) 相對路徑但本身就存在：直接用（例如 "./data/val/host/host.wav"）
    if os.path.exists(item):
        return item

    # 3) 只剩檔名：跟 root 拼起來
    return os.path.join(root, os.path.basename(item))

def _list_wavs(folder: str):
    if folder is None:
        return []
    return sorted(glob.glob(os.path.join(folder, "*.wav")))


def _rms_normalize(wav: torch.Tensor, target_rms: float) -> torch.Tensor:
    """
    將音訊正規化到指定的 RMS 響度。

    Args
        wav        : (C, L) tensor，値域 [-1, 1]
        target_rms : 目標 RMS（建議 0.1，即 -20 dBFS）。0.0 表示不處理。
    Returns
        wav 經幅度縮放後的 tensor，clamp 到 [-1, 1]
    """
    if target_rms <= 0.0:
        return wav
    rms = wav.pow(2).mean().sqrt()
    if rms < 1e-8:          # 靜音片段，跳過防止除以零
        return wav
    return torch.clamp(wav * (target_rms / rms), -1.0, 1.0)


class InvASNetAudioPairDataset(Dataset):
    """
    Returns (cover(host_music), secret(speech)) as tensors:
      cover:  (1, segment_length)
      secret: (1, segment_length)

    If folders are empty, falls back to synthetic audio so the pipeline can run.
    """
    def __init__(self, mode: str = "train"):
        self.mode = mode
        self.num_secrets = int(getattr(c, "num_secrets", 1))
        if mode == "train":
            self.host_files    = _list_wavs(getattr(c, "TRAIN_HOST_PATH",    None))
            self.secret_files  = _list_wavs(getattr(c, "TRAIN_SECRET_PATH",  None))
            self.secret2_files = (_list_wavs(getattr(c, "TRAIN_SECRET2_PATH", None))
                                  if self.num_secrets == 2 else [])
            self.host_root     = getattr(c, "TRAIN_HOST_PATH",    "")
            self.secret_root   = getattr(c, "TRAIN_SECRET_PATH",  "")
            self.secret2_root  = getattr(c, "TRAIN_SECRET2_PATH", "")
        else:
            self.host_files    = _list_wavs(getattr(c, "VAL_HOST_PATH",    None))
            self.secret_files  = _list_wavs(getattr(c, "VAL_SECRET_PATH",  None))
            self.secret2_files = (_list_wavs(getattr(c, "VAL_SECRET2_PATH", None))
                                  if self.num_secrets == 2 else [])
            self.host_root     = getattr(c, "VAL_HOST_PATH",    "")
            self.secret_root   = getattr(c, "VAL_SECRET_PATH",  "")
            self.secret2_root  = getattr(c, "VAL_SECRET2_PATH", "")

        self.seg_len  = int(getattr(c, "segment_length", 44160))
        self.channels = int(getattr(c, "channels_in", 1))
        self.secret_target_rms = float(getattr(c, "secret_target_rms", 0.0))
        if self.secret_target_rms > 0.0:
            print(f"[Dataset/{mode}] secret RMS 正規化: target_rms={self.secret_target_rms:.3f} ({20*__import__('math').log10(self.secret_target_rms):.1f} dBFS)")
        else:
            print(f"[Dataset/{mode}] secret RMS 正規化: 關閉（保持原始音量）")

        # 如果發現找不到檔案，它會自動開啟「合成模式 (_make_synth)」
        need_files = [self.host_files, self.secret_files]
        if self.num_secrets == 2:
            need_files.append(self.secret2_files)
        self.use_synth = any(len(f) == 0 for f in need_files)

        self.n_pairs = (min(len(f) for f in need_files)
                        if not self.use_synth else 1_000_000)

    def __len__(self):
        return self.n_pairs

    def _make_synth(self):
        # Simple synthetic signals: (channels, L)
        t = torch.linspace(0, 1, self.seg_len).unsqueeze(0).repeat(self.channels, 1)
        cover   = torch.randn(self.channels, self.seg_len) * 0.02 + 0.05 * torch.sin(2 * torch.pi * 220.0 * t)
        secret1 = torch.randn(self.channels, self.seg_len) * 0.02 + 0.05 * torch.sin(2 * torch.pi * 440.0 * t)
        cover   = torch.clamp(cover,   -1.0, 1.0)
        secret1 = torch.clamp(secret1, -1.0, 1.0)
        secret1 = _rms_normalize(secret1, self.secret_target_rms)   # 將合成 secret 正規化
        if self.num_secrets == 2:
            secret2 = torch.randn(self.channels, self.seg_len) * 0.02 + 0.05 * torch.sin(2 * torch.pi * 880.0 * t)
            secret2 = torch.clamp(secret2, -1.0, 1.0)
            secret2 = _rms_normalize(secret2, self.secret_target_rms)
            return cover, secret1, secret2
        return cover, secret1

    def __getitem__(self, idx):
        if self.use_synth:
            return self._make_synth()
        #print("[DEBUG] TRAIN_HOST_PATH =", c.TRAIN_HOST_PATH,
      #"files =", os.listdir(c.TRAIN_HOST_PATH) if os.path.exists(c.TRAIN_HOST_PATH) else None)

        #print("[DEBUG] TRAIN_SECRET_PATH =", c.TRAIN_SECRET_PATH, "files =", os.listdir(c.TRAIN_SECRET_PATH) if os.path.exists(c.TRAIN_SECRET_PATH) else None)

        # Real audio mode (to be implemented when you have wav files)
        # 當程式讀到真正的 .wav 檔案時，它會對聲音進行「強迫症般」的標準化處理：
        def _load_wav_mono_44k(path, target_sr=44100, target_len=16384):
            wav, sr = torchaudio.load(path)          # wav: (C, T)
            # 強制轉單聲道，如果你的音檔是左右聲道（立體聲），它會把它們平均起來，變成單聲道。
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)  # stereo -> mono

            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)    # 強制統一音質，如果音檔的採樣率不是設定的 44100，它會自動幫你轉檔。

            # pad / crop to fixed length
            # 強制裁切長度 (最重要！)，神經網路一次只能處理固定長度的資料（我們在 config 裡設定的 segment_length = 44160）。
            # 如果音檔太短，它會補上靜音（0）來湊滿長度 (pad)。
            # 如果音檔太長（例如一首 3 分鐘的歌），它會隨機切一段 44160 長度的片段出來 (torch.randint)。這不僅解決了長度問題，還順便做到了「資料擴增（Data Augmentation）」，讓模型每次都聽到同一首歌的不同段落，變相增加訓練資料！
            T = wav.shape[1]
            if T < target_len:
                wav = torch.nn.functional.pad(wav, (0, target_len - T))
            else:
                start = torch.randint(0, T - target_len + 1, (1,)).item()
                wav = wav[:, start:start + target_len]
            return wav

        # 在 __getitem__(self, idx) 裡用：
        host_item   = self.host_files[idx % len(self.host_files)]
        secret_item = self.secret_files[idx % len(self.secret_files)]

        host_path   = host_item   if os.path.exists(host_item)   else os.path.join(self.host_root,   host_item)
        secret_path = secret_item if os.path.exists(secret_item) else os.path.join(self.secret_root, secret_item)

        cover   = _load_wav_mono_44k(host_path,   target_sr=c.host_sr, target_len=c.segment_length)
        secret1 = _load_wav_mono_44k(secret_path, target_sr=c.host_sr, target_len=c.segment_length)
        secret1 = _rms_normalize(secret1, self.secret_target_rms)   # 將 secret 正規化到目標 RMS

        if self.num_secrets == 2:
            secret2_item = self.secret2_files[idx % len(self.secret2_files)]
            secret2_path = (secret2_item if os.path.exists(secret2_item)
                            else os.path.join(self.secret2_root, secret2_item))
            secret2 = _load_wav_mono_44k(secret2_path, target_sr=c.host_sr, target_len=c.segment_length)
            secret2 = _rms_normalize(secret2, self.secret_target_rms)
            return cover, secret1, secret2

        return cover, secret1
    

# DataLoaders
trainloader = DataLoader(
    InvASNetAudioPairDataset(mode="train"),
    batch_size=getattr(c, "batch_size", 16),
    shuffle=True,
    pin_memory=True,
    num_workers=0,   # Windows ��ĳ���� 0�A�קK multiprocessing ���D
    drop_last=False
)

testloader = DataLoader(
    InvASNetAudioPairDataset(mode="val"),
    batch_size=getattr(c, "batchsize_val", 2),
    shuffle=False,
    pin_memory=True,
    num_workers=0,
    drop_last=True
)


