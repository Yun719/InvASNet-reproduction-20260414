>檔案：`rrdb_denselayer_1d.py` 在系統中的角色：**INV_block 的子網路（φ, ρ, η 三個函數的實作）**

## 含註解程式

```python
import torch
import torch.nn as nn


class ResidualDenseBlock_out_1D(nn.Module):
    """
    1D version of ResidualDenseBlock_out:
    Conv2d -> Conv1d, kernel=3, padding=1 to keep length.
    """
    def __init__(self, input_ch, output_ch, bias=True):
        super().__init__()
        #                      輸入通道      輸出通道 核大小 步長  填充
        self.conv1 = nn.Conv1d(input_ch,          32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv1d(input_ch + 32,     32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv1d(input_ch + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv1d(input_ch + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv1d(input_ch + 4 * 32, output_ch, 3, 1, 1, bias=bias)
        #                                         ↑ 最後一層輸出你要的通道數

        self.lrelu = nn.LeakyReLU(inplace=True)  # 激活函數（帶負斜率）

        # ⭐ 關鍵初始化：最後一層全設為零
        nn.init.zeros_(self.conv5.weight)
        if self.conv5.bias is not None:
            nn.init.zeros_(self.conv5.bias)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))                          # 第 1 層
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))      # 第 2 層，看到 x 和 x1
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))  # 第 3 層，看到 x, x1, x2
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1))) # 第 4 層，看到全部
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))     # 第 5 層（輸出層，無激活）
        return x5

```

## 傳入參數

在 `invblock.py` 有這段程式

[invblock.py 說明筆記](invblock.py_說明筆記.md)

```python
# ρ  
self.r = subnet_constructor(self.split_len1, self.split_len2)  
# η  
self.y = subnet_constructor(self.split_len1, self.split_len2)  
# φ  
self.f = subnet_constructor(self.split_len2, self.split_len1)  
```

對應到 `ResidualDenseBlock_out_1D` 在 `__init__()` 的 `input_ch`、 `output_ch`

接著 `invblock.py` 這段的 `self.f(x2)` 、 `self.r(y1)`、 `self.y(y1)` 

```python
t2 = self.f(x2)  
y1 = x1 + t2  
s1, t1 = self.r(y1), self.y(y1)  
y2 = self.e(s1) * x2 + t1  
```

裡面的 `x2` 、 `y1`  實際上會傳到 `ResidualDenseBlock_out_1D` 的 `forward(self, x)` 中 `forward` 的 `x` 就是它！
 
## 一. 這個檔案是做什麼的？

在 `invblock.py` 有這段程式

[invblock.py 說明筆記](invblock.py_說明筆記.md)

```python
# ρ  
self.r = subnet_constructor(self.split_len1, self.split_len2)  
# η  
self.y = subnet_constructor(self.split_len1, self.split_len2)  
# φ  
self.f = subnet_constructor(self.split_len2, self.split_len1) 

...

if not rev:  
    t2 = self.f(x2)  
    y1 = x1 + t2  
    s1, t1 = self.r(y1), self.y(y1)  
    y2 = self.e(s1) * x2 + t1
```


`r` 、 `y` 、`f` 指的就是這份程式 `ResidualDenseBlock_out_1D` 本身，它的作用是

>**接收一段特徵張量，輸出一段轉換後的特徵張量**

它本身不做隱寫，只是一個能從輸入中提取特徵的「函數模塊」。 你可以把它想像成：把一段音訊特徵「消化理解」後，輸出對應的「操作參數」。

在整個隱寫過程中，因為：
- $\phi$：負責處理 $x_2$，並將結果加到 $x_1$ 上。（`self.f`）
- $\rho$：負責計算**縮放 (Scale)** 係數。（`self.r`）
- $\eta$：負責計算**平移 (Translation)** 係數。（`self.y`）

這些參數一開始會是隨機值，我們需要透過訓練來改變他們，這份程式就是為了訓練他們而誕生的

## 二. Conv1d 參數逐一解析

```
nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias)
```

| 參數             | 這裡的值                       | 意義                      |
| -------------- | -------------------------- | ----------------------- |
| `in_channels`  | 每層不同（見下表）                  | 輸入通道數                   |
| `out_channels` | 32（前 4 層）/ output_ch（最後一層） | 輸出通道數                   |
| `kernel_size`  | 3                          | 每次看左右各 1 個相鄰時間步         |
| `stride`       | 1                          | 每次移動 1 步（不跳格）           |
| `padding`      | 1                          | 兩端各補 1 個零，確保輸出長度 = 輸入長度 |
>**kernel=3, stride=1, padding=1 的效果**： 輸入長度 L → 輸出長度 L，**長度不變** 
公式
$$L_{out} = \frac{L + 2 \times \text{padding} - \text{kernel}}{\text{stride}} + 1
= \frac{L + 2 - 3}{1} + 1
= L$$

~~這公式不太重要，看看就好~~


## 三. 稠密連接（Dense Connection）原理

普通 CNN（資訊只往前傳）

```
x → Conv1 → x1 → Conv2 → x2 → Conv3 → x3
```

每一層只能看到「前一層」的輸出。

DenseNet（每一層都能看到所有前面的輸出）

```
x ──────────────────────────────────→ Conv1 → x1
x, x1 ──────────────────────────────→ Conv2 → x2
x, x1, x2 ──────────────────────────→ Conv3 → x3
x, x1, x2, x3 ──────────────────────→ Conv4 → x4
x, x1, x2, x3, x4───────────────────→ Conv5 → 輸出
```

**好處**：
- 每一層都能直接讀取「原始輸入」和「所有中間特徵」
- 梯度更容易往回傳，訓練更穩定（緩解梯度消失）
- 特徵重用，模型更有效率

##  四. 資料流維度追蹤

以 `input_ch=2, output_ch=2` 為例（實際的 INV_block 呼叫值）：

| 步驟    | 輸入張量                 | 輸入通道              | 輸出通道 | 輸出張量             |
| ----- | -------------------- | ----------------- | ---- | ---------------- |
| 輸入    | `x`                  | 2                 | —    | `(B, 2, L)`      |
| conv1 | `x`                  | 2                 | 32   | `x1: (B, 32, L)` |
| conv2 | `cat(x, x1)`         | 2+32=**34**       | 32   | `x2: (B, 32, L)` |
| conv3 | `cat(x,x1,x2)`       | 2+32+32=**66**    | 32   | `x3: (B, 32, L)` |
| conv4 | `cat(x,x1,x2,x3)`    | 2+32+32+32=**98** | 32   | `x4: (B, 32, L)` |
| conv5 | `cat(x,x1,x2,x3,x4)` | 2+32×4=**130**    | 2    | `x5: (B, 2, L)`  |

>**長度 L 從頭到尾不變**，這是 padding=1 的功勞。 **輸出通道 = output_ch = 2**，與輸入通道相同，方便後續做加法/乘法。

### `(B , C , L)` 這 3 個數值的意義是

>對應到上面的 `(B, 2, L)` ，因為 `(B , C , L)` 很常出現，所以特別筆記一下

1. `B` = Batch Size (批次大小)
	- **白話文：** 你這次**同時**丟了幾筆資料給機器算。
	- **舉例：** 假設你一次讓電腦同時處理 8 段音訊，那麼 `B` 就是 8。神經網路為了提升 GPU 平行運算的效率，不會一次只算一筆，而是把很多筆綁在一起同時算。

2. `C` = Channels (通道數 / 特徵厚度)
	- **白話文：** 這筆資料有幾層「屬性」或「特徵」。 
	- **舉例：** * 如果是立體聲音樂，通常會有左右聲道，所以通道數是 2（訓練資料的通道數是 1 ）。
	    - **對應你的程式碼：** 還記得你前面用 `x.narrow` 切出了 `x2` 嗎？你設定了 `split_len2 = 1 * 2 = 2`。所以這裡的 `2` 就是你切出來的「語音」特徵，它剛好具有 2 個通道的厚度（也就是我們上一回討論到的 `input_ch`）。

3. `L` = Length (序列長度 / 時間維度)
	- **白話文：** 這段資料在時間軸上有多長。
	- **舉例：** 假設你的音訊採樣率是 16,000 Hz，並且這段音訊有 1 秒鐘，那麼 `L` 就是 16,000。這代表這段特徵在時間上有 16,000 個資料點。

## 五. 零初始化的設計原因 ⭐

```python
# ⭐ 關鍵初始化：最後一層全設為零
nn.init.zeros_(self.conv5.weight)
if self.conv5.bias is not None:
	nn.init.zeros_(self.conv5.bias)
```

**如果不這樣做：** 訓練開始時，隨機初始化的 conv5 輸出一個很大的隨機值， INV_block 的縮放因子 `e(s) = exp(...)` 就會變成一個很大的數， 造成 **數值爆炸（NaN / Inf）**，模型直接廢掉。

**這樣做之後：**

訓練初期：conv5 輸出 ≈ 0

```
訓練初期：conv5 輸出 ≈ 0
→ f(x2) ≈ 0  →  y1 = x1 + 0 = x1
→ e(r(y1)) ≈ exp(0) = 1，η(y1) ≈ 0  →  y2 = 1 × x2 + 0 = x2
→ 整個 INV_block ≈ 恆等函數（不改變任何東西）
```

模型從「什麼都不做」慢慢學習「如何改變」，訓練非常穩定。

## 七. LeakyReLU 的選擇

```python
self.lrelu = nn.LeakyReLU(inplace=True)
```

|激活函數|負數輸入|優點|
|---|---|---|
|ReLU|直接輸出 0|簡單|
|**LeakyReLU**|**輸出一個小值（預設斜率 0.01）**|**負數不會完全「死掉」**|

>`inplace=True` 代表直接在原記憶體上修改，節省 GPU 記憶體。 注意：最後一層 conv5 **沒有** LeakyReLU，因為輸出需要可正可負（無限制範圍）。

## 八. 如何修改這個模組

### 8.1 修改中間層的寬度（32 → 其他值）

```python
# 把所有 32 改成 64
HIDDEN = 64   # ← 新增這行，然後全部換成 HIDDEN

self.conv1 = nn.Conv1d(input_ch,              HIDDEN, 3, 1, 1, bias=bias)
self.conv2 = nn.Conv1d(input_ch + HIDDEN,     HIDDEN, 3, 1, 1, bias=bias)
self.conv3 = nn.Conv1d(input_ch + 2 * HIDDEN, HIDDEN, 3, 1, 1, bias=bias)
self.conv4 = nn.Conv1d(input_ch + 3 * HIDDEN, HIDDEN, 3, 1, 1, bias=bias)
self.conv5 = nn.Conv1d(input_ch + 4 * HIDDEN, output_ch, 3, 1, 1, bias=bias)

```

>⚠️ 增大 HIDDEN 會增加參數量，訓練更慢但表達能力更強。


### 8.2 增加層數（4 層 → 5 層）

```python
HIDDEN = 32
self.conv1 = nn.Conv1d(input_ch,              HIDDEN, 3, 1, 1, bias=bias)
self.conv2 = nn.Conv1d(input_ch + HIDDEN,     HIDDEN, 3, 1, 1, bias=bias)
self.conv3 = nn.Conv1d(input_ch + 2 * HIDDEN, HIDDEN, 3, 1, 1, bias=bias)
self.conv4 = nn.Conv1d(input_ch + 3 * HIDDEN, HIDDEN, 3, 1, 1, bias=bias)
self.conv5 = nn.Conv1d(input_ch + 4 * HIDDEN, HIDDEN, 3, 1, 1, bias=bias)  # ← 改：不再是輸出層
self.conv6 = nn.Conv1d(input_ch + 5 * HIDDEN, output_ch, 3, 1, 1, bias=bias)  # ← 新增

# forward 裡也要對應修改：
def forward(self, x):
    x1 = self.lrelu(self.conv1(x))
    x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
    x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
    x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
    x5 = self.lrelu(self.conv5(torch.cat((x, x1, x2, x3, x4), 1)))  # ← 改：加激活
    x6 = self.conv6(torch.cat((x, x1, x2, x3, x4, x5), 1))           # ← 新增
    return x6

```

>⚠️ 記得把最後一層（conv6）的 zero init 也加上。

### 8.3 更換激活函數

```python
# 原本
self.lrelu = nn.LeakyReLU(inplace=True)

# 可以換成 GELU（更現代，類似 Transformer 的選擇）
self.lrelu = nn.GELU()

# 或 ELU
self.lrelu = nn.ELU(inplace=True)
```

>⚠️ GELU 不支援 `inplace=True`，記得移除。

### 8.4 修改 kernel size（感受野）

```python
# 原本：kernel=3，只看左右各 1 個時間步
nn.Conv1d(input_ch, 32, kernel_size=3, stride=1, padding=1)

# 改成 kernel=5，看左右各 2 個時間步（更大範圍）
nn.Conv1d(input_ch, 32, kernel_size=5, stride=1, padding=2)
# 規則：padding = (kernel_size - 1) // 2  → 確保長度不變

```