
import torch.nn as nn
import config as c
from invblock import INV_block

# -------------------------------------------------------
# 根據 config.haar_levels 動態決定每個 INV_block 的 channel 分割數
#
# 1D Haar DWT 每疊一次：C channels → 2C channels (low + high)
# haar_levels 次後，cover_d / secret_d 各有 channels_in * 2^haar_levels 個 channel
#
# INV_block 以 harr=True 建立時：split_len = in_1 * 2
# 因此：in_1 = channels_in * 2^haar_levels / 2 = channels_in * 2^(haar_levels-1)
#
# 例：haar_levels=1, channels_in=1 → _in=1, split_len=2  (原始架構)
#     haar_levels=3, channels_in=1 → _in=4, split_len=8  (3次Haar)
# -------------------------------------------------------
class Hinet(nn.Module):

    def __init__(self):
        super(Hinet, self).__init__()

        _haar        = int(getattr(c, "haar_levels",  1))
        _ch          = int(getattr(c, "channels_in",  1))
        _num_secrets = int(getattr(c, "num_secrets",  1))

        # cover 側 per channel（harr=True: split_len = in_1 * 2）
        _in_cover  = _ch * (2 ** (_haar - 1))
        # secret 側 per channel（所有秘密的通道數合計再 / 2）
        _in_secret = _ch * (2 ** (_haar - 1)) * _num_secrets

        self.inv1  = INV_block(in_1=_in_cover, in_2=_in_secret)
        self.inv2  = INV_block(in_1=_in_cover, in_2=_in_secret)
        self.inv3  = INV_block(in_1=_in_cover, in_2=_in_secret)
        self.inv4  = INV_block(in_1=_in_cover, in_2=_in_secret)
        self.inv5  = INV_block(in_1=_in_cover, in_2=_in_secret)
        self.inv6  = INV_block(in_1=_in_cover, in_2=_in_secret)
        self.inv7  = INV_block(in_1=_in_cover, in_2=_in_secret)
        self.inv8  = INV_block(in_1=_in_cover, in_2=_in_secret)

        self.inv9  = INV_block(in_1=_in_cover, in_2=_in_secret)
        self.inv10 = INV_block(in_1=_in_cover, in_2=_in_secret)
        self.inv11 = INV_block(in_1=_in_cover, in_2=_in_secret)
        self.inv12 = INV_block(in_1=_in_cover, in_2=_in_secret)
        self.inv13 = INV_block(in_1=_in_cover, in_2=_in_secret)
        self.inv14 = INV_block(in_1=_in_cover, in_2=_in_secret)
        self.inv15 = INV_block(in_1=_in_cover, in_2=_in_secret)
        self.inv16 = INV_block(in_1=_in_cover, in_2=_in_secret)

    def forward(self, x, rev=False):

        if not rev:
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
            out = self.inv16(x, rev=True)
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
            out = self.inv1(out, rev=True)

        return out

