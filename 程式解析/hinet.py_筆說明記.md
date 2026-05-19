>檔案：`hinet.py` 在系統中的角色：**把 16 個 INV_block 串成一條流水線，負責整個模型的正向（隱藏）與反向（還原）流程**

## 一、這個檔案在整個系統的哪裡？

```
datasets.py ← 讀取音檔
↓
train.py ← 訓練主程式
↓
model.py ← 最外層封裝 (Model)
↓
hinet.py ← ⭐ 你現在在這裡（堆疊 16 個 INV_block）
↓
invblock.py ← 每一層的可逆耦合運算
↓
rrdb_denselayer_1d.py ← 每個耦合層內部的特徵提取子網路
```

## 二、完整程式碼 + 註解版

```python
import torch.nn as nn
from invblock import INV_block   # 引入可逆積木

class Hinet(nn.Module):

    def __init__(self):
        super(Hinet, self).__init__()

        # ── 建立 16 個獨立的可逆積木 ──────────────────────────────
        # 每一個 inv 都有自己的一組參數（權重不共享）
        # 可以把它們想像成 16 位各自獨立學習的「魔術師」
        self.inv1  = INV_block()
        self.inv2  = INV_block()
        self.inv3  = INV_block()
        self.inv4  = INV_block()
        self.inv5  = INV_block()
        self.inv6  = INV_block()
        self.inv7  = INV_block()
        self.inv8  = INV_block()

        self.inv9  = INV_block()
        self.inv10 = INV_block()
        self.inv11 = INV_block()
        self.inv12 = INV_block()
        self.inv13 = INV_block()
        self.inv14 = INV_block()
        self.inv15 = INV_block()
        self.inv16 = INV_block()

    def forward(self, x, rev=False):

        if not rev:
            # ── 正向：隱藏模式（把語音藏進音樂）──────────────────
            # 資料從 inv1 → inv2 → ... → inv16，一層一層往前推
            out = self.inv1(x)
            out = self.inv2(out)
            out = self.inv3(out)
            out = self.inv4(out)
            out = self.inv5(out)
            out = self.inv6(out)
            out = self.inv7(out)
            out = self.inv8(out)

            out = self.inv9(out)
            out = self.inv10(out)
            out = self.inv11(out)
            out = self.inv12(out)
            out = self.inv13(out)
            out = self.inv14(out)
            out = self.inv15(out)
            out = self.inv16(out)

        else:
            # ── 反向：還原模式（從 stego 取出語音）───────────────
            # ⭐ 注意：每層呼叫時傳入 rev=True，並且順序「完全顛倒」
            out = self.inv16(x, rev=True)   # 最後一層先跑
            out = self.inv15(out, rev=True)
            out = self.inv14(out, rev=True)
            out = self.inv13(out, rev=True)
            out = self.inv12(out, rev=True)
            out = self.inv11(out, rev=True)
            out = self.inv10(out, rev=True)
            out = self.inv9(out, rev=True)

            out = self.inv8(out, rev=True)
            out = self.inv7(out, rev=True)
            out = self.inv6(out, rev=True)
            out = self.inv5(out, rev=True)
            out = self.inv4(out, rev=True)
            out = self.inv3(out, rev=True)
            out = self.inv2(out, rev=True)
            out = self.inv1(out, rev=True)  # 第一層最後跑

        return out

```

## 三、核心概念：為什麼反向要「倒序」？

這是可逆神經網路最重要的一個概念。

**請想像一個上鎖的箱子，鑰匙有 16 把：**

```
【上鎖過程（正向）】
箱子 → 鎖1 → 鎖2 → 鎖3 → ... → 鎖16 → 上鎖完成

【開鎖過程（反向）】
開鎖完成 ← 開鎖16 ← 開鎖15 ← ... ← 開鎖2 ← 開鎖1 ← 箱子
```

看起來挺簡單的吧！

你只能用「最後鎖上的那把鎖的鑰匙」先開，然後一把一把倒回去。  
如果開鎖順序搞錯，箱子會毀掉（資訊還原失敗）。

**用公式表達：**

```
正向結果：y = inv16( inv15( ... inv2( inv1(x) ) ... ) )

反向：x = inv1⁻¹( inv2⁻¹( ... inv15⁻¹( inv16⁻¹(y) ) ... ) )
```

每個 `inv_k⁻¹` 就是呼叫 `self.inv_k(data, rev=True)`。

程式裡的 `out = self.inv15(out, rev=True)` 這段，意思是把 `(out, rev=True)` 傳給 `invblock.py` 的 `forward(self, x, rev=False)`

[inv_block.py說明筆記](invblock.py_說明筆記)

## 四、資料流維度追蹤

Hinet 前後維度完全不變：

```
輸入 x：(B, 4C, L/2)

↓ inv1 → inv2 → ... → inv16

輸出 out：(B, 4C, L/2) ← 維度完全相同！
```

>每個 INV_block 輸入和輸出維度相同，所以串多少層都不影響形狀。  
以預設值 B=2, C=1, L=44160 為例，資料全程都是 **(2, 4, 22080)**。

關於維度的說明，可以看 [rrdb 說明筆記 (B,C,L) 的部分](rrdb_denselayer_1d.py_說明筆記)

## 五、16 層的視覺化流程圖

### 正向（隱藏）

```
x (含 cover + secret 的特徵)
│
├─ inv1  ─→ 微調一次
├─ inv2  ─→ 再微調
├─ inv3  ─→ ...
│    ...（共 16 次漸進式轉換）
└─ inv16 ─→ 最終輸出 y

y 的前半部分 = stego（偽裝音樂）
y 的後半部分 = z（訓練時丟棄，推論時用隨機雜訊取代）
```

### 反向（還原）

```
stego + 隨機雜訊
│
├─ inv16(rev) ─→ 倒退一步
├─ inv15(rev) ─→ 再倒退
│    ...（共 16 次逆向還原）
└─ inv1(rev)  ─→ 最終輸出 x̂

x̂ 的後半部分 = 還原出的 secret（語音）
```

## 六、如何修改這個模組

### 6.1 增加或減少層數

假設想改成 8 層：

```python
def __init__(self):
    super(Hinet, self).__init__()
    # 只保留 8 個
    self.inv1 = INV_block()
    self.inv2 = INV_block()
    # ... 到 inv8

def forward(self, x, rev=False):
    if not rev:
        out = self.inv1(x)
        out = self.inv2(out)
        # ... 到 inv8
    else:
        out = self.inv8(x, rev=True)
        # ... 倒回到 inv1
```

>⚠️ **正向和反向的層數必須完全一致，反向順序必須完全倒序**，否則無法還原。

### 6.2 想驗證反向能否完美還原（Debug 用）

```python
import torch
from hinet import Hinet

net = Hinet()
net.eval()

x = torch.randn(1, 4, 22080)     # 隨機輸入
y = net(x, rev=False)             # 正向
x_hat = net(y, rev=True)          # 反向

# 計算誤差，理論上應該接近 0
error = (x - x_hat).abs().max()
print(f"最大還原誤差: {error:.2e}")  # 正常應該 < 1e-5
```

## 七、常見問題

**Q：為什麼 16 個 `INV_block()` 要分開宣告，而不是用 `nn.ModuleList`？**
> 功能上完全相同，用 `nn.ModuleList` 更簡潔，但原版程式直接列出來更直觀，  
> 且方便在某層做特殊設定（如果未來需要的話）。

**Q：每個 `INV_block` 的參數是共享的嗎？**
> **不是。** 每個 `self.inv1`、`self.inv2`... 都是獨立建立的物件，  
> 擁有各自獨立的神經網路權重。它們「架構相同，但學到的東西不同」。

**Q：正向輸出的 `out` 是 stego 嗎？**
> 不完全是。`out` 是頻率域的特徵（經 DWT 後的資料）。  
> 需要在 `train.py` 中先 `.narrow` 取前半，再通過 `IWT` 才能還原成 stego 波形。



















