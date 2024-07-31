import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)


# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return x * self.sigmoid(x)


# class ChannelAttention(nn.Module):
#     def __init__(self, inplanes):
#         super(ChannelAttention, self).__init__()
#         self.max_pool = nn.MaxPool2d(1)
#         self.avg_pool = nn.AvgPool2d(1)
#         # 通道注意力，即两个全连接层连接
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_channels=inplanes, out_channels=inplanes // 16, kernel_size=1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=inplanes // 16, out_channels=inplanes, kernel_size=1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         max_out = self.fc(self.max_pool(x))
#         avg_out = self.fc(self.avg_pool(x))
#         # 最后输出的注意力应该为非负
#         out = self.sigmoid(max_out + avg_out)
#         return out


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=7 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 压缩通道提取空间信息
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 经过卷积提取空间注意力权重
        x = torch.cat([max_out, avg_out], dim=1)
        out = self.conv1(x)
        # 输出非负
        out = self.sigmoid(out)
        return out


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        # self.sa = SpatialAttention(kernel_size)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


# torch.Size([1, 64, 512, 512])
# torch.Size([1, 1, 512, 512])