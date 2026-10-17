import torch
import torch.nn as nn

from ultralytics.nn.modules.block import C2f
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import PSABlock

# 引入你自己的 RepViTblock
from .RepViTblock import RepViTblock


class C3k2_RepViTBlock(C2f):
    """
    C3k2 + RepViTBlock + PSA Attention
    """

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=True,
        g=1, 
        e=0.5,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)

        # 内部block替换
        self.m = nn.ModuleList(
            nn.Sequential(
                RepViTblock(
                    in1=self.c,
                    inp=self.c,
                    hidden_dim=2 * self.c,   # ⚠️必须
                    oup=self.c,
                    kernel_size=3,
                    stride=1,               # ⚠️必须=1
                    use_se=1,
                    use_hs=0
                ),
                PSABlock(
                    self.c,
                    attn_ratio=0.5,
                    num_heads=max(self.c // 64, 1)
                )
            )
            for _ in range(n)
        )