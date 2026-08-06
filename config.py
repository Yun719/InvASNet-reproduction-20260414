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
channels_in  = 1       # 代表你的聲音是「單聲道 (Mono)」。如果是立體聲這裡就要改成 2。
haar_levels  = 3       # ✨ DWT/IWT 疊加次數。
                       # 1 → 原始架構（2C channel, L/2 長度）
                       # 3 → 3 次 Haar（8C channel, L/8 長度），需重新訓練，不能沿用舊權重
quantize_simulation = False  # ✨ 是否啟用量化模擬（Quantization-Aware Training）
                             # True  → 訓練時模擬 16-bit WAV 量化誤差，讓模型更耐 WAV 存取的精度損失
                             # False → 不加量化噪聲，訓練速度稍快，適合初期實驗
num_secrets = 1             # ✨ 要藏幾段秘密音訊（1 或 2）
                             # 1 → 原始單秘密架構（網路輸入 4C）
                             # 2 → 雙秘密架構（網路輸入 6C），需重新訓練，不能沿用舊權重
segment_length = 44160 # 不管你的音檔是一首歌還是一句話，程式都會把它強制隨機截取成 44160 個採樣點的長度。
                       # ⚠️ 須能被 2^haar_levels 整除。44160 / 8 = 5520，haar_levels=3 時 OK。
host_sr   = 44100      # 「載體聲音 (Host/Cover)」預期是高音質的 44.1 kHz（CD 音質）
secret_sr = 16000      # 「秘密聲音 (Secret)」預期是 16 kHz（一般的語音對話音質）
secret_target_rms = 0.01 # ✨ 嵌入前將 secret 正規化到此 RMS 響度（− 40 dBFS ≈ 0.01）
                         #    越小 = stego 高頻雜音越少（建議範圍 0.01~0.05）
                         #    0.0 = 關閉，保持原始音量

# ---- training hyperparams ----
log10_lr = -4.5         # 學習率 (Learning Rate)。也就是教練每次巴 AI 頭的「力道」。這裡設定為 $10^{-4.5}$（大約 0.0000316），這是一個在訓練可逆神經網路時非常常見且安全的「微調力道」。
lr = 10 ** log10_lr
epochs = 100         # ✨總共要讓 AI 訓練幾個日夜輪迴。

betas = (0.5, 0.999)
weight_decay = 1e-5
weight_step = 1000
gamma = 0.5

lamda_reconstruction = 5    # 還原秘密的權重。設定為 5 代表我們非常看重「秘密能不能完美拿出來」，這項不准出錯。
lamda_guide = 1             # 偽裝聲音的權重（時域 MSE）
lamda_low_frequency = 1     # 低頻約束的權重

# ---- psychoacoustic perceptual loss ----
lamda_psy       = 1.0   # ✨ 心理聲學感知損失權重（0.0 = 關閉，不影響原始訓練流程）
                        #    推薦調整範圍：0.5 ~ 2.0
psy_n_fft       = 4096  #    每幀 FFT 點數（建議 2 的次方；須 <= segment_length=44160）
psy_alpha       = 0.05  #    遮蔽比例：越小 = 門限越嚴格（推薦 0.05 ~ 0.20）
psy_spread_width = 31   #    Bark 域擴散核寬度（奇數，越大 = 遮蔽範圍越廣）

batch_size = 2          # 每次教練丟「幾題」給 AI 寫。因為聲音資料很佔記憶體，所以這裡設 2（一次餵兩組聲音進去）。如果你顯示卡記憶體夠大，可以調成 4 或 8，訓練會更快。
batchsize_val = 1
shuffle_val = False
val_freq = 0        # 多久考一次試 ?

# ---- checkpoint ----
MODEL_PATH = os.path.join(os.getcwd(), "model") + os.sep
SAVE_freq = 5

suffix = "model_checkpoint_00000.pt"     # ✨接著做的模型檔名
tain_next = False      # ✨如果你今天訓練到第 10 個 Epoch 關掉電腦，明天想繼續，就把這個改成 True，它就會去 model 資料夾底下讀取 model.pt 繼續跑。
trained_epoch = 0       # ✨上次跑到第幾輪 ?（重跑請填 0）

# ---- dataset paths (audio) ----
INVASN_DATA_ROOT = "./data"  # 這代表所有的資料都要放在你目前這個程式碼資料夾裡面，一個名為 data 的子資料夾下。

TRAIN_HOST_PATH    = os.path.join(INVASN_DATA_ROOT, "train", "host")
TRAIN_SECRET_PATH  = os.path.join(INVASN_DATA_ROOT, "train", "secret")
TRAIN_SECRET2_PATH = os.path.join(INVASN_DATA_ROOT, "train", "secret2")  # num_secrets=2 時使用
VAL_HOST_PATH      = os.path.join(INVASN_DATA_ROOT, "val",   "host")
VAL_SECRET_PATH    = os.path.join(INVASN_DATA_ROOT, "val",   "secret")
VAL_SECRET2_PATH   = os.path.join(INVASN_DATA_ROOT, "val",   "secret2")  # num_secrets=2 時使用

# ---- misc ----
silent = False
progress_bar = True    # 要不要顯示進度條?
live_visualization = False
loss_display_cutoff = 2.0
loss_names = ["L", "lr"]

