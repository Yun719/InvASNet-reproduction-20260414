# InvASNet 程式說明筆記

> 適合對象：本組組員，具備基礎 Python / PyTorch 知識即可閱讀。

---

## 一、這個程式在做什麼？

**InvASNet** 是一種「**音訊隱寫術 (Audio Steganography)**」系統。

> 隱寫術 ≠ 加密。加密讓人知道「有秘密但看不懂」；隱寫術讓人根本「不知道有秘密」。

| 角色               | 說明          | 例子         |
| ---------------- | ----------- | ---------- |
| **Cover（掩護音訊）**  | 表面上聽起來正常的音樂 | 一首流行歌      |
| **Secret（秘密音訊）** | 你想偷偷傳遞的聲音   | 一段對話語音     |
| **Stego（偽裝音訊）**  | 藏完後對外公開的音樂  | 聽起來還是那首流行歌 |

**目標：**
- 🔒 **隱藏**：把 Secret 偷偷藏進 Cover，產生出聽起來幾乎一樣的 Stego
- 🔓 **還原**：從 Stego 裡把 Secret 完整拿回來

---

## 二、整體架構圖

```
【資料輸入】
  cover.wav  ──→  DWT  ──→  cover_d (B, 2C, L/2)  ──┐
  secret.wav ──→  DWT  ──→  secret_d(B, 2C, L/2)  ──┤ torch.cat
                                                       ↓
                                              x (B, 4C, L/2)
                                                       │
【模型主體】                                            ↓
                                          ┌─────────────────────┐
                                          │  Hinet (16 層)       │
                                          │  inv1 → inv2 → ...  │
                                          │  每層是一個 INV_block│
                                          └──────────┬──────────┘
                                                     ↓
                                              y (B, 4C, L/2)
                                                     │
                              ┌──────────────────────┤
                              ↓                      ↓
                         y_steg (前半)          y_z (後半)
                              │                （丟掉，換高斯雜訊）
                              ↓
【輸出】                    IWT
                              ↓
                        stego.wav  ← 對外公開的偽裝音訊
```

---

## 三、每個檔案的職責

| 檔案 | 職責 | 比喻 |
|---|---|---|
| `config.py` | 所有超參數設定 | 食譜上的份量說明 |
| `datasets.py` | 讀取 `.wav` 檔、裁切成固定長度 | 食材預處理 |
| `modules/dwt1d.py` | 1D Haar 小波轉換 | 把音訊「分頻」 |
| `rrdb_denselayer_1d.py` | 子網路（特徵提取器） | 模型的「大腦小零件」 |
| `invblock.py` | 可逆耦合層（INN 的核心） | 可完美逆轉的魔術 |
| `hinet.py` | 把 16 個 INV_block 堆疊起來 | 16 層魔術疊加 |
| `model.py` | 包裝 Hinet 成最終模型 | 最外層的盒子 |
| `train.py` | 訓練流程 | 教 AI 怎麼做的過程 |
| `app.py` | Gradio 網頁介面 | 使用者操作介面 |

---

## 四、關鍵模組詳解

### 4.1 DWT1D — 小波轉換（`modules/dwt1d.py`）

**作用**：把一段音訊從「時域」分解成「低頻」和「高頻」兩部分，同時長度縮短一半。

```
輸入: x (B, C, L)
  ↓ 把偶數位置和奇數位置分開
low  = (x_even + x_odd) / 2   ← 低頻（整體輪廓）
high = (x_even - x_odd) / 2   ← 高頻（細節/噪音）
輸出: [low, high] (B, 2C, L/2)
```

**IWT1D** 是它的完美逆操作：
```
x_even = low + high
x_odd  = low - high
→ 交錯重組回 (B, C, L)，資訊零損失
```

> **為什麼要先做 DWT？**
> 小波轉換把音訊分頻後，神經網路可以在「頻率域」操作，
> 更容易在不影響低頻（音樂主旋律）的情況下，把秘密藏在高頻細節裡。

---

### 4.2 ResidualDenseBlock_out_1D — 子網路（`rrdb_denselayer_1d.py`）

**作用**：INV_block 裡用來提取特徵的「函數」，負責把輸入轉換成對應的縮放參數和平移參數。

**架構（Dense Connection）**：

```
x ──→ Conv1 ──→ x1
x, x1 ──→ Conv2 ──→ x2
x, x1, x2 ──→ Conv3 ──→ x3
x, x1, x2, x3 ──→ Conv4 ──→ x4
x, x1, x2, x3, x4 ──→ Conv5 ──→ 輸出
```

每一層都能看到「所有前面層的輸出」，資訊流動非常豐富。

> **重要細節**：`Conv5` 的初始權重設為 **全零**。
> 這確保訓練剛開始時，網路的輸出接近 0，
> 即 INV_block 在初始時近似「恆等變換（不做任何事）」，
> 讓訓練更穩定，不會一開始就爆炸。

---

### 4.3 INV_block — 可逆耦合層（`invblock.py`）⭐ 最核心

這是整個系統的心臟。它的魔法是：**無論怎麼轉換，都能完美逆轉還原。**

#### 輸入切分

```python
x1 = x.narrow(1, 0, split_len1)           # 前半 = cover 的特徵
x2 = x.narrow(1, split_len1, split_len2)  # 後半 = secret 的特徵
```

#### 正向（隱藏）

| 步驟 | 公式 | 說明 |
|---|---|---|
| 1 | `y1 = x1 + f(x2)` | cover 加上由 secret 計算出的偏移量 |
| 2 | `y2 = e(r(y1)) × x2 + η(y1)` | secret 做仿射變換（縮放 + 平移），參數由 cover 決定 |

#### 反向（還原）

| 步驟 | 公式 | 說明 |
|---|---|---|
| 1 | `y2 = (x2 - η(x1)) / e(r(x1))` | 先把 secret 還原（做反仿射） |
| 2 | `y1 = x1 - f(y2)` | 再把 cover 還原（做反加法） |

#### 非對稱設計的原因

```
cover 側：只做「加減法」（Additive Coupling）
  → cover 的音量幅度不會被縮放，stego 聽起來和 cover 一樣

secret 側：做「乘法 + 加法」（Affine Coupling）
  → 表達能力更強，能更完整地把資訊藏進去
```

#### e(s) 是什麼？

```python
def e(self, s):
    v = self.clamp * 2 * (torch.sigmoid(s) - 0.5)  # 把 s 壓縮到 [-clamp, clamp]
    v = torch.clamp(v, -10.0, 10.0)                 # 防止 exp overflow
    return torch.exp(v)
```

sigmoid 把任意數字壓縮到一個安全範圍，再取 exp，確保縮放因子永遠是正數且不爆炸。

---

### 4.4 Hinet — 堆疊 16 層（`hinet.py`）

把 16 個 INV_block 串聯起來。

**正向**：`x → inv1 → inv2 → ... → inv16 → y`

**反向**：`y → inv16(rev) → ... → inv1(rev) → x`

反向時必須**完全倒序**，這是可逆網路的基本要求。

---

### 4.5 損失函數（`train.py`）

訓練時同時優化三個目標：

| 損失 | 計算方式 | 權重 | 意義 |
|---|---|---|---|
| `g_loss` | `MSE(stego, cover)` | λ=3 | Stego 要聽起來像 Cover |
| `r_loss` | `MSE(recovered_secret, secret)` | λ=5 | 秘密要能完整還原 ⭐最重要 |
| `l_loss` | `MSE(stego_low_freq, cover_low_freq)` | λ=1 | Stego 的低頻（主旋律）要和 Cover 一樣 |

```
Total Loss = 5 × r_loss + 3 × g_loss + 1 × l_loss
```

---

## 五、完整資料流（以訓練為例）

```
Step 1: 讀取音訊
  cover.wav (44100 Hz) ──→ 裁切成 44160 個採樣點
  secret.wav           ──→ 裁切成相同長度

Step 2: DWT 分頻
  cover  (B, 1, 44160) ──→ DWT ──→ cover_d  (B, 2, 22080)
  secret (B, 1, 44160) ──→ DWT ──→ secret_d (B, 2, 22080)

Step 3: 拼接
  x = [cover_d | secret_d] → (B, 4, 22080)
        └─前半──┘ └─後半──┘

Step 4: 正向過模型（隱藏）
  y = Hinet(x)  → (B, 4, 22080)
  y_steg = y[:, :2, :]   ← 前半（對應 cover）
  y_z    = y[:, 2:, :]   ← 後半（對應 secret，訓練時丟掉）

Step 5: IWT 還原波形
  stego = IWT(y_steg)  → (B, 1, 44160)

Step 6: 反向過模型（還原）
  z_rand = 隨機高斯雜訊（模擬未知的 secret 分佈）
  x_hat = Hinet([y_steg | z_rand], rev=True)
  secret_hat = IWT(x_hat[:, 2:, :])

Step 7: 計算 Loss 並反向傳播更新權重
```

> **為什麼反向時用隨機雜訊取代 secret？**
> 因為使用者在提取時不知道原來的 secret，
> 所以訓練時就故意模擬這個情況，讓模型學會即使後半是雜訊也能從前半（stego）裡把 secret 挖出來。

---

## 六、推論介面（`app.py`）

使用 Gradio 建立網頁，分兩個功能：

### 🔒 隱藏（Hide）
```
cover.wav + secret.wav
    ↓ DWT + cat + Hinet(正向) + IWT
stego.wav（存檔輸出）
```

### 🔓 提取（Extract）
```
stego.wav
    ↓ DWT
steg_d (B, 2, L/2)
    ↓ 拼上隨機雜訊
[steg_d | z_rand] (B, 4, L/2)
    ↓ Hinet(反向)
    ↓ 取後半 + IWT
recovered_secret.wav（存檔輸出）
```

可選濾波器：高通 300Hz + 低通 3400Hz，用來去除人聲以外的雜音。

---

## 七、設定檔速查（`config.py`）

| 參數 | 值 | 說明 |
|---|---|---|
| `channels_in` | 1 | 單聲道 |
| `segment_length` | 44160 | 每段音訊固定長度（約 1 秒） |
| `host_sr` | 44100 | Cover 採樣率（CD 音質） |
| `secret_sr` | 16000 | Secret 採樣率（語音品質） |
| `batch_size` | 2 | 每次訓練的批次大小 |
| `epochs` | 20 | 訓練總輪數 |
| `lamda_reconstruction` | 5 | 還原 Secret 的損失權重（最高） |
| `lamda_guide` | 3 | Stego 像 Cover 的損失權重 |
| `clamp` | 2.0 | INV_block 中 e(s) 的縮放上限 |
| `init_scale` | 0.01 | 初始化權重的縮放比例 |

---

## 八、常見問題

**Q: 為什麼提取出來的聲音有雜訊？**
> 因為提取時用的是「隨機雜訊」填入後半，模型必須靠 stego 前半的資訊來還原 secret。如果訓練不足，或訓練 loss 設計不合理，secret 資訊可能沒被有效「編碼」進 stego 的前半部分。

**Q: 為什麼 cover 側只用加法，不用乘法？**
> 乘法（exp 縮放）會改變音量幅度，讓 stego 和 cover 聽起來音量差很多，容易被察覺。只用加法代表只疊加微小擾動，更隱蔽。

**Q: 16 層 INV_block 是怎麼決定的？**
> 這是從原始 HiNet（圖像版）搬過來的設計。層數越多，模型表達能力越強，但訓練越慢。這是個超參數，可以調整。
