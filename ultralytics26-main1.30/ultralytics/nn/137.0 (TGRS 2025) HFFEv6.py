import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 0. 工具函数与标准 CBR 模块
# ==========================================
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class CBR(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# ==========================================
# 1. LSKblock：大感受野选择性空间注意力（净化复杂背景）
# ==========================================
class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + \
               attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn

# ==========================================
# 2. EMA：高效多尺度注意力机制（处理多尺度抖动）
# ==========================================
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g, c//g, h, w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

# ==========================================
# 3. 终极完全体 HFFE (HF-IGate-Pro)
# ==========================================
class HFFE(nn.Module):
    def __init__(self, c1, c2, k=3):
        super(HFFE, self).__init__()
        feature_low_channel = c1[0]  
        feature_high_channel = c1[1]

        # 1) 统一通道对齐层
        self.conv_low_1x1 = nn.Conv2d(feature_low_channel, c2, 1, bias=False)
        self.conv_high_1x1 = nn.Conv2d(feature_high_channel, c2, 1, bias=False)

        # 2) 换装 LSKblock：取代普通 SpatialAttention，在边缘提取前净化杂波
        self.low_sa = LSKblock(c2)
        self.high_sa = LSKblock(c2)

        # 3) 可学习的高频动态抑制因子（初始设为 0.1，允许网络动态压制背景噪声）
        self.hf_scale_low = nn.Parameter(torch.tensor([0.1]))
        self.hf_scale_high = nn.Parameter(torch.tensor([0.1]))

        # 4) 双向交互门控映射
        self.gate_l2h = nn.Sequential(
            nn.Conv2d(c2 * 2, c2, 1, bias=False),
            nn.Sigmoid()
        )
        self.gate_h2l = nn.Sequential(
            nn.Conv2d(c2 * 2, c2, 1, bias=False),
            nn.Sigmoid()
        )

        # 5) 换装 EMA：取代老旧的 CoordAttention，完美应对小目标尺度骤变
        self.ema = EMA(c2)

        # 6) 最终融合输出
        self.conv_final = CBR(c2 * 2, c2, 1)

    def _high_freq(self, x):
        """高效的通道分离式拉普拉斯高频边缘提取算子"""
        b, c, h, w = x.size()
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=x.dtype, device=x.device)
        kernel = kernel.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
        return F.conv2d(x, kernel, padding=1, groups=c)

    def forward(self, x):
        x_low, x_high = x
        
        # 1) 分辨率双线性对齐
        if x_low.shape[-2:] != x_high.shape[-2:]:
            x_high = F.interpolate(x_high, size=x_low.shape[-2:], mode="bilinear", align_corners=True)

        # 2) 通道降维/升维对齐到目标 c2
        low = self.conv_low_1x1(x_low)
        high = self.conv_high_1x1(x_high)

        low_res = low
        high_res = high

        # 3) LSK 前置净化（大核过滤环境底噪）
        low_sa = self.low_sa(low)
        high_sa = self.high_sa(high)

        # 4) 自适应动态边缘增强
        low_hf = low_sa + self.hf_scale_low * self._high_freq(low_sa)
        high_hf = high_sa + self.hf_scale_high * self._high_freq(high_sa)

        # 5) 双向门控特征交叉调制
        gate_l2h = self.gate_l2h(torch.cat([high_hf, low_hf], dim=1))
        gate_h2l = self.gate_h2l(torch.cat([low_hf, high_hf], dim=1))

        low_gated = low_res * gate_h2l
        high_gated = high_res * gate_l2h

        # 6) 特征汇聚与 EMA 空间-通道级跨维度多尺度注意交互
        out = self.conv_final(torch.cat([low_gated, high_gated], dim=1))
        out = self.ema(out)  # EMA 内部已自带残差乘算机制

        return out