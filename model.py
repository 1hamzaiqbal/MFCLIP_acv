import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientAttention(nn.Module):
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super(EfficientAttention, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, key_channels, head_count, value_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.attention = EfficientAttention(out_channels, key_channels, head_count, value_channels)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        residual = self.skip_conv(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        out += residual
        return self.activation(out)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.activation(self.bn(self.conv(x)))
        return x


class UNetLikeGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=32, n_blocks=3):
        super(UNetLikeGenerator, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        # 初始卷积层
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 下采样层
        self.down1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
            ResBlock(ngf * 2, ngf * 2, ngf // 2, 8, ngf * 2)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            ResBlock(ngf * 4, ngf * 4, ngf, 8, ngf * 4)
        )

        # 中间的ResBlock
        self.mid_blocks = nn.ModuleList([
            ResBlock(ngf * 4, ngf * 4, ngf, 8, ngf * 4) for _ in range(n_blocks)
        ])

        # 上采样层
        self.up2 = UpBlock(ngf * 4, ngf * 2)
        self.up1 = UpBlock(ngf * 2, ngf)

        # 最终输出层
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
        )

    def forward(self, x):
        # 初始特征提取
        x1 = self.initial(x)  # 尺寸不变: [3, 224, 224] -> [64, 224, 224]

        # 下采样
        x2 = self.down1(x1)  # [64, 224, 224] -> [128, 112, 112]
        x3 = self.down2(x2)  # [128, 112, 112] -> [256, 56, 56]

        # 中间ResBlock
        out = x3
        for block in self.mid_blocks:
            out = block(out)  # 尺寸保持不变: [256, 56, 56]

        # 上采样
        out = self.up2(out)  # [256, 56, 56] -> [128, 112, 112]
        out = out + x2  # 跳跃连接, 尺寸: [128, 112, 112]
        out = self.up1(out)  # [128, 112, 112] -> [64, 224, 224]
        out = out + x1  # 跳跃连接, 尺寸: [64, 224, 224]

        # 最终输出
        out = self.final(out)  # [64, 224, 224] -> [3, 224, 224]

        return out


# 测试代码
generator = UNetLikeGenerator()
input_tensor = torch.randn(1, 3, 224, 224)
output = generator(input_tensor)
print(output.shape)  # 应输出 torch.Size([1, 3, 224, 224])