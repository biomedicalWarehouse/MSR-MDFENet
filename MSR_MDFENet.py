import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import torch.fft as fft
from collections import OrderedDict
from torch import Tensor

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

# 旨在通过显式地建模通道间的依赖关系来提升网络的表示能力
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class FeatureEnhancedAttention(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.feature_weights = nn.Linear(emb_size, emb_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, emb_size, time_steps, spatial_dim = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(batch_size * time_steps * spatial_dim, emb_size)
        weights = self.sigmoid(self.feature_weights(x))
        weights = weights.view(batch_size, time_steps, spatial_dim, emb_size)
        x = x.view(batch_size, time_steps, spatial_dim, emb_size)
        enhanced_features = x * weights
        enhanced_features = enhanced_features.permute(0, 3, 1, 2)
        return enhanced_features

class MultiScaleResidualBlockDWSK(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(MultiScaleResidualBlockDWSK, self).__init__()
        mid_channels = in_channels // 4
        mid_channels_sk = out_channels // 4
        self.conv1x1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv1x1sk = nn.Conv2d(in_channels, mid_channels_sk, kernel_size=1, stride=stride, bias=False)
        self.bn1sk = nn.BatchNorm2d(mid_channels_sk)
        self.dropout = nn.Dropout2d(0.3)
        self.convDW3x3 = DepthwiseSeparableConv2d(mid_channels, mid_channels * 2, kernel_size=3, stride=1, padding=1)
        self.bnDW3x3 = nn.BatchNorm2d(mid_channels * 2)
        self.convDW5x5 = DepthwiseSeparableConv2d(mid_channels, mid_channels * 2, kernel_size=5, stride=1, padding=2)
        self.bnDW5x5 = nn.BatchNorm2d(mid_channels * 2)

        self.mish = Mish()
        self.se = SEBlock(in_channels)
        self.feature_enhanced_attention = FeatureEnhancedAttention(in_channels)
        self.shortcut = nn.Sequential()
        self.catt = CoordAtt(out_channels, out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x,):
        out = self.conv1x1(x)
        out = self.mish(self.bn1(out))
        out3x3 = self.convDW3x3(out)
        out3x3 = self.mish(self.bnDW3x3(out3x3))
        out5x5 = self.convDW5x5(out)
        out5x5 = self.mish(self.bnDW5x5(out5x5))
        out = torch.cat([out3x3, out5x5], dim=1)
        outSE = self.se(out)
        outFIA = self.feature_enhanced_attention(out)
        out = torch.cat([outSE, outFIA], dim=1)
        out = self.catt(out)
        out += self.shortcut(x)
        return out

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=16):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        # self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        # self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        # self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.conv1 = DepthwiseSeparableConv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.conv2 = DepthwiseSeparableConv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = DepthwiseSeparableConv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(mip)
        # self.relu = h_swish()
        self.relu = Swish()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y

class FeatureExtraction(nn.Module):
    def __init__(self, in_channels=1, out_channel=64):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, kernel_size=(1, 5), stride=(1, 2)),
            nn.BatchNorm2d(out_channel),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            nn.Dropout(0.5),
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.shallownet(x)
        return x

class FeatureEnhancement(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.shallownet = nn.Sequential(
            MultiScaleResidualBlockDWSK(in_channels, in_channels*2),  # 自定义残差块
            nn.BatchNorm2d(in_channels*2),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            nn.Dropout(0.5),
        )
        self.shallownet2 = nn.Sequential(
            MultiScaleResidualBlockDWSK(in_channels * 2, in_channels * 4),  # 自定义残差块
            nn.BatchNorm2d(in_channels * 4),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            nn.Dropout(0.3),
        )
        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.depthwise_separable_conv = DepthwiseSeparableConvolution(4 * in_channels, 8 * in_channels)  # 确保维度一致
    def forward(self, x):
        x = self.shallownet(x)
        x = self.shallownet2(x)
        x = self.projection(x)
        x = x.permute(0, 2, 1)
        x = self.depthwise_separable_conv(x)
        x = x.permute(0, 2, 1)
        return x

class ClassificationHead(nn.Module):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.reduction = Reduce('b n e -> b e', reduction='mean')
        self.norm = nn.LayerNorm(emb_size)
        self.fc = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        x = self.reduction(x)
        x = self.norm(x)
        x = self.fc(x)
        return x

class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.3):
        super(DepthwiseSeparableConvolution, self).__init__()
        self.depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        self.pointwise_bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.Mish = Mish()
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = F.relu(x)
        x = self.pointwise_bn(x)
        x = self.dropout(x)
        return x

class MSR_MDFENet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, out_channel=256, out_channel1=256):
        super().__init__()
        self.feature_extraction = FeatureExtraction(in_channels, out_channel=out_channel)
        self.feature_enhancement = FeatureEnhancement(out_channel)
        self.classification_head = ClassificationHead(8*out_channel1, n_classes)
    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.feature_enhancement(x)
        x = self.classification_head(x)
        return x


if __name__ == "__main__":
    batch_size = 8
    channel = 1
    time = 1280
    num_classes = 2
    img_size = 18

    input_data = torch.randn(batch_size,img_size, time)

    model = MSR_MDFENet(in_channels=1, n_classes=2)
    outputs = model(input_data)
    print(outputs.shape)
