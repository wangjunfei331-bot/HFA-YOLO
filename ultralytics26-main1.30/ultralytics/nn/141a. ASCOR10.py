import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision

class SpatialContextGating(nn.Module):
    """保持高效的 3x3 小核空间门控"""
    def __init__(self, kernel_size=3):
        super(SpatialContextGating, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(y))

class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
        super(DCNv2, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.offset_channels =  3 * kernel_size * kernel_size
        
        self.conv_offset_mask = nn.Conv2d(
            in_channels, 
            self.offset_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=True
        )
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, dilation=dilation, 
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.conv_offset_mask.weight, 0)
        nn.init.constant_(self.conv_offset_mask.bias, 0)

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1).contiguous()
        mask = torch.sigmoid(mask).contiguous()
        
        x = torchvision.ops.deform_conv2d(
            input=x, weight=self.conv.weight, offset=offset, mask=mask,
            bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation
        )
        x = self.bn(x)
        x = self.act(x)
        return x

class GaussianPriorBranch(nn.Module):
    """
    源自 TCSVT 2025 LEGNet 的高斯平滑分支
    利用固定的高斯核进行深度可分离卷积，过滤高频噪声，提取稳定的目标中心高斯分布概率。
    """
    def __init__(self, dim, size=5, sigma=1.0):
        super().__init__()
        # 生成高斯核
        gaussian = self.gaussian_kernel(size, sigma)
        
        # 将其转化为不可训练的常量 Parameter (requires_grad=False)
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=size//2, groups=dim, bias=False)
        self.gaussian.weight = nn.Parameter(gaussian.repeat(dim, 1, 1, 1), requires_grad=False)
        
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.SiLU(inplace=True)

    def gaussian_kernel(self, size: int, sigma: float):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
             for y in range(-size // 2 + 1, size // 2 + 1)
             ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()

    def forward(self, x):
        if self.gaussian.weight.requires_grad:
            self.gaussian.weight.requires_grad = False
            
        # 使用 .detach() 确保该权重绝对不参与当前计算图的反向传播
        # 这就彻底避免了 YOLO 底层引擎的强行覆盖
        gaussian_weight = self.gaussian.weight.detach()
        
        # 使用 F.conv2d 直接调用算子，避开 nn.Conv2d 的状态追踪
        edges_o = F.conv2d(
            x, 
            gaussian_weight, 
            bias=None, 
            stride=self.gaussian.stride, 
            padding=self.gaussian.padding, 
            dilation=self.gaussian.dilation, 
            groups=self.gaussian.groups
        )
        
        return self.act(self.norm(edges_o))

class ASCOR(nn.Module):
    """
    ASCOR-G (Gaussian Augmented)
    融合了 TCSVT 2025 的高斯先验机制。
    - 分支 A: 局部细粒度感知 (3x3 DW)
    - 分支 B: DCNv2 自适应形变感知
    - 分支 C: 串联致密大感受野
    - 分支 D (New): Gaussian 平滑先验 (零参数计算，抗噪稳场)
    """
    def __init__(self, in_channels, out_channels=None, reduction=2):
        super(ASCOR, self).__init__()
        self.out_channels = out_channels if out_channels else in_channels
        mid_channels = in_channels // reduction
        
        # 1. 降维层
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True)
        )
        
        # --- 四异构特征感知矩阵 ---
        self.branch_local = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True)
        )
        
        self.branch_dcn = DCNv2(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3)
        
        self.branch_dense = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True)
        )
        
        
        self.branch_gaussian = GaussianPriorBranch(dim=mid_channels, size=3, sigma=0.5)
        
        # 恢复 3 分支融合的维度
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(mid_channels * 3, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True)
        )
        
        # 3. 投影升维
        self.expand_conv = nn.Sequential(
            nn.Conv2d(mid_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels)
        )
        
        # 4. 残差流
        if in_channels != self.out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.spatial_gating = SpatialContextGating(kernel_size=3)
        self.gauss_scale = nn.Parameter(torch.zeros(1, mid_channels, 1, 1))

    def forward(self, x):
        identity = self.shortcut(x)
        identity = self.spatial_gating(identity)
        
        feat_reduced = self.reduce_conv(x)
        
        # 并发进入四大分支
        feat_local = self.branch_local(feat_reduced)
        feat_dcn = self.branch_dcn(feat_reduced)
        feat_dense = self.branch_dense(feat_reduced)
        
        
        # 通道级拼接，结合所有异构特征
        feat_concat = torch.cat([feat_local, feat_dcn, feat_dense], dim=1)
        feat_aggregated = self.fuse_conv(feat_concat)

        feat_gauss = self.branch_gaussian(feat_aggregated)
        feat_refined = feat_aggregated + self.gauss_scale * feat_gauss
        
        out = self.expand_conv(feat_refined)
        final_out = F.silu(out + identity)
        
        if final_out.shape[2:] != x.shape[2:]:
            final_out = F.interpolate(final_out, size=x.shape[2:], mode='nearest')
            
        return final_out.contiguous()