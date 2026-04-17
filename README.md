# 架構

```commandline
audio_steganography_project/
├── data/                        # 存放所有音訊資料的總資料夾 (由 config.py 設定)
│   ├── train/                   # 訓練集
│   │   ├── host/                # 載體音訊 (負責掩護的普通聲音)
│   │   └── secret/              # 秘密音訊 (要被藏起來的聲音)
│   └── val/                     # 驗證/測試集
│       ├── host/                # 測試用的載體音訊
│       └── secret/              # 測試用的秘密音訊
├── model/                       # 存放訓練結果的資料夾
│   └── model.pt                 # 訓練好的模型權重檔 (由 train.py 定期存檔)
│
├── config.py                    # 📍 控制中心：設定所有路徑、學習率、裁切長度等超參數
├── datasets.py                  # 🥦 備料流水線：讀取 wav 檔、處理防呆、裁切長度、轉單聲道
│
├── train.py                     # 👨‍💼 訓練主程式：負責把資料餵給模型、計算 Loss、執行更新 (optimizer.step)
├── model.py                     # 📦 模型包裝盒：負責初始化參數，並提供正反向切換開關 (rev=True/False)
├── hinet.py                     # 🚇 可逆網路主幹：16 道關卡的雙向隧道 (串接所有 INV_block)
├── invblock.py                  # 🪄 核心代數魔法：仿射耦合層，計算加減乘除以實現完美可逆
├── rrdb_denselayer_1d.py        # 🧑‍🍳 更新函數黑盒子：產生轉換係數的 1D 卷積神經網路 (可以換成 U-Net)
│
├── README.md                    # 專案說明檔 (記錄怎麼啟動程式、修改了哪些架構)
│
└── auto_push.bat                # 用來方便 push 到 Github 的 Windows 腳本批次檔
```

# 訓練用資料集

>需要自行建立 `data/` 資料夾，再將訓練用資料放入，請參考**架構**建置資料夾

+ 載體音樂：[GTZAN Genre Collection](https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection)
+ 隱藏語音：[LibriSpeech](https://www.openslr.org/12)
  + 選擇 `dev-clean.tar.gz` 
+ 在 windows 環境下，在檔案總管的搜尋欄輸入 `*.flac` 可以一次找到資料夾下（包含子資料夾）的 `*.flac` 檔案

## 轉換格式

因為資料集的檔案格式程式不吃（程式只吃 `.wav`），所以需要使用 `FFmpeg` 轉成 `.wav` 格式，這裡提供 Windows `CMD` 轉換指令

```bash
for /R %f in (*.flac *.au) do ffmpeg -i "%f" "%~dpnf.wav"
```
`FFmpeg` 安裝可以參考[雲彩的 blog](https://yuncolorblog.com/posts/%E6%8A%80%E8%A1%93%E7%AD%86%E8%A8%98/download-music-from-youtube/) 或 [Ivon 的 Blog](https://ivonblog.com/posts/yt-dlp-installation/)

這行指令可以將 `.flac` 和 `.au` 格式轉換成 `.wav` 格式



刪除 `.flac` 和 `.au` `CMD` 用指令：

刪除前請務必確認 `.wav` 可以正常撥放！

```bash
del *.flac *.au
```

---

# 必要套件

python 版本：3.12.4 （其他版本不確定行不行）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

```bash
pip install numpy scipy tqdm matplotlib tensorboardx soundfile 
```



---

# 參數設定
和參數有關的設定都在 `config.py` 中

比較重要的有
- **batch_size**：每幾筆資料更新一次權重
- **epochs**：訓練次數
- **tain_next**：是否讀取上次的訓練進度
- **progress_bar**：要不要顯示訓練的進度條