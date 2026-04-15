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
├──  rrdb_denselayer_1d.py        # 🧑‍🍳 更新函數黑盒子：產生轉換係數的 1D 卷積神經網路 (可以換成 U-Net)
│
├── requirements.txt             # 套件依賴清單 (例如：torch, torchaudio)
├──  README.md                    # 專案說明檔 (記錄怎麼啟動程式、修改了哪些架構)
│
└── auto_push.bat                # 用來方便 push 到 Github 的 Windows 腳本批次檔
```