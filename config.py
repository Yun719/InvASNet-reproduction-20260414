# =========================
# InvASNet (Audio, 1D) config
# =========================

import os
clamp = 2.0
init_scale = 0.01
# ---- device / run ----
device_ids = [0]              # �u�� CUDA �h�d�~�|�Ψ�
checkpoint_on_error = True      # 萬一跑到一半當機，立刻幫你存檔

# ---- audio basic ----
channels_in = 1               # 代表你的聲音是「單聲道 (Mono)」。如果是立體聲這裡就要改成 2。
segment_length = 44160    # 不管你的音檔是一首歌還是一句話，程式都會把它強制隨機截取成 44160 個採樣點 的長度。
host_sr = 44100         # 採樣率 (Sample Rate)。這代表「載體聲音 (Host/Cover)」預期是高音質的 44.1 kHz（CD 音質）
secret_sr = 16000   # 「秘密聲音 (Secret)」預期是 16 kHz（一般的語音對話音質）

# ---- training hyperparams ----
log10_lr = -4.5         # 學習率 (Learning Rate)。也就是教練每次巴 AI 頭的「力道」。這裡設定為 $10^{-4.5}$（大約 0.0000316），這是一個在訓練可逆神經網路時非常常見且安全的「微調力道」。
lr = 10 ** log10_lr
epochs = 20         # 總共要讓 AI 訓練幾個日夜輪迴。

betas = (0.5, 0.999)
weight_decay = 1e-5
weight_step = 1000
gamma = 0.5

lamda_reconstruction = 5    # 還原秘密的權重。設定為 5 代表我們非常看重「秘密能不能完美拿出來」，這項不准出錯。
lamda_guide = 1     # 偽裝聲音的權重。只要聽起來像就好。
lamda_low_frequency = 1 # 低頻約束的權重

batch_size = 2          # 每次教練丟「幾題」給 AI 寫。因為聲音資料很佔記憶體，所以這裡設 2（一次餵兩組聲音進去）。如果你顯示卡記憶體夠大，可以調成 4 或 8，訓練會更快。
batchsize_val = 1
shuffle_val = False
val_freq = 1

# ---- checkpoint ----
MODEL_PATH = os.path.join(os.getcwd(), "model") + os.sep
SAVE_freq = 1

suffix = "model.pt"
tain_next = False       # 如果你今天訓練到第 10 個 Epoch 關掉電腦，明天想繼續，就把這個改成 True，它就會去 model 資料夾底下讀取 model.pt 繼續跑。
trained_epoch = 0

# ---- dataset paths (audio) ----
INVASN_DATA_ROOT = "./data"  # 這代表所有的資料都要放在你目前這個程式碼資料夾裡面，一個名為 data 的子資料夾下。

TRAIN_HOST_PATH   = os.path.join(INVASN_DATA_ROOT, "train", "host")     # 把你要當作「掩護」的普通聲音（訓練用）放這裡。
TRAIN_SECRET_PATH = os.path.join(INVASN_DATA_ROOT, "train", "secret")   # 把你要「藏起來」的秘密聲音（訓練用）放這裡。
VAL_HOST_PATH     = os.path.join(INVASN_DATA_ROOT, "val", "host")
VAL_SECRET_PATH   = os.path.join(INVASN_DATA_ROOT, "val", "secret")

# ---- misc ----
silent = False
progress_bar = False
live_visualization = False
loss_display_cutoff = 2.0
loss_names = ["L", "lr"]

