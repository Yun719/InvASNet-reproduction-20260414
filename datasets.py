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


class InvASNetAudioPairDataset(Dataset):
    """
    Returns (cover(host_music), secret(speech)) as tensors:
      cover:  (1, segment_length)
      secret: (1, segment_length)

    If folders are empty, falls back to synthetic audio so the pipeline can run.
    """
    def __init__(self, mode: str = "train"):
        self.mode = mode
        if mode == "train":
            self.host_files = _list_wavs(getattr(c, "TRAIN_HOST_PATH", None))
            self.secret_files = _list_wavs(getattr(c, "TRAIN_SECRET_PATH", None))
        else:
            self.host_files = _list_wavs(getattr(c, "VAL_HOST_PATH", None))
            self.secret_files = _list_wavs(getattr(c, "VAL_SECRET_PATH", None))
        if mode == "train":
            self.host_root = c.TRAIN_HOST_PATH
            self.secret_root = c.TRAIN_SECRET_PATH
        else:
            self.host_root = c.VAL_HOST_PATH
            self.secret_root = c.VAL_SECRET_PATH

        self.seg_len = int(getattr(c, "segment_length", 44160))
        self.channels = int(getattr(c, "channels_in", 1))

        # If no real audio, we'll use synthetic fallback
        self.use_synth = (len(self.host_files) == 0) or (len(self.secret_files) == 0)

        # For real audio mode later, we will require 1-to-1 pairing by index.
        self.n_pairs = min(len(self.host_files), len(self.secret_files)) if not self.use_synth else 1000000
        #print("[DEBUG] host_files[0] =", self.host_files[0])

    def __len__(self):
        return self.n_pairs

    def _make_synth(self):
        # Simple synthetic signals: (1, L)
        # cover: low-amplitude noise + slow sine
        noted = torch.randn(self.channels, self.seg_len) * 0.02
        t = torch.linspace(0, 1, self.seg_len).unsqueeze(0).repeat(self.channels, 1)
        cover = noted + 0.05 * torch.sin(2 * torch.pi * 220.0 * t)

        # secret: slightly different sine + noise
        secret = (torch.randn(self.channels, self.seg_len) * 0.02 +
                  0.05 * torch.sin(2 * torch.pi * 440.0 * t))

        # clamp to [-1, 1] like normalized audio
        cover = torch.clamp(cover, -1.0, 1.0)
        secret = torch.clamp(secret, -1.0, 1.0)
        return cover, secret

    def __getitem__(self, idx):
        if self.use_synth:
            return self._make_synth()
        #print("[DEBUG] TRAIN_HOST_PATH =", c.TRAIN_HOST_PATH,
      #"files =", os.listdir(c.TRAIN_HOST_PATH) if os.path.exists(c.TRAIN_HOST_PATH) else None)

        #print("[DEBUG] TRAIN_SECRET_PATH =", c.TRAIN_SECRET_PATH, "files =", os.listdir(c.TRAIN_SECRET_PATH) if os.path.exists(c.TRAIN_SECRET_PATH) else None)

        # Real audio mode (to be implemented when you have wav files)
        # For now, we keep it strict so you notice if you accidentally think it's reading files.
        def _load_wav_mono_44k(path, target_sr=44100, target_len=16384):
            wav, sr = torchaudio.load(path)          # wav: (C, T)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)  # stereo -> mono

            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)

            # pad / crop to fixed length
            T = wav.shape[1]
            if T < target_len:
                wav = torch.nn.functional.pad(wav, (0, target_len - T))
            else:
                start = torch.randint(0, T - target_len + 1, (1,)).item()
                wav = wav[:, start:start + target_len]
            return wav

        # 在 __getitem__(self, idx) 裡用：
        host_item = self.host_files[idx % len(self.host_files)]
        secret_item = self.secret_files[idx % len(self.secret_files)]

        # 如果 item 本身已經是完整/可用路徑，就直接用；不然才 join 資料夾
        host_path = host_item if os.path.exists(host_item) else os.path.join(self.host_root, host_item)
        secret_path = secret_item if os.path.exists(secret_item) else os.path.join(self.secret_root, secret_item)



        cover = _load_wav_mono_44k(host_path, target_sr=c.host_sr, target_len=c.segment_length)
        secret = _load_wav_mono_44k(secret_path, target_sr=c.host_sr, target_len=c.segment_length)

        return cover, secret
    

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


