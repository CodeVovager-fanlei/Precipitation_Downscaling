import torch
from torch import nn
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm


# 自定义权重初始化函数
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights."""
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=3):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.attention = CBAMBlock(num_feat)  # 加入 CBAM 注意力机制

        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        out = x5 * 0.2 + x  # 残差连接
        return self.attention(out)  # 加入注意力机制


# 修改后的 ResLap 网络
class ResLap(nn.Module):
    def __init__(self, num_feat=64):
        super(ResLap, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, 32, kernel_size=3, padding=1)

        # 使用 nn.ModuleList 包装
        self.rdbs = nn.ModuleList([
            ResidualDenseBlock(32, 32),
            ResidualDenseBlock(32, 32),
            ResidualDenseBlock(32, 32)
        ])

        # 使用 nn.ModuleList 包装
        self.dconv = nn.ModuleList([
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=3, padding=1),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.Upsample(size=(100, 159), mode='bilinear', align_corners=False)
        ])

        # 使用 nn.ModuleList 包装
        self.conv2 = nn.ModuleList([nn.Conv2d(32, 3, kernel_size=3, padding=1) for i in range(3)])
        self.res_construction = nn.ModuleList([
            nn.ConvTranspose2d(num_feat, 3, kernel_size=5, stride=3, padding=1),
            nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1),
            nn.Upsample(size=(100, 159), mode='bilinear', align_corners=False)
        ])

        # 初始化权重
        default_init_weights([self.conv1, *self.conv2])

    def forward(self, x):
        x1 = self.conv1(x)

        x1 = self.rdbs[0](x1)
        x1 = self.dconv[0](x1)
        r1 = self.conv2[0](x1)
        f1 = self.res_construction[0](x)
        f1 = r1 + f1
        f1 = torch.cat((torch.sigmoid(f1[:, 0, :, :]).unsqueeze(dim=1), f1[:, 1:, :, :]), dim=1)

        x2 = self.rdbs[1](x1)
        x2 = self.dconv[1](x2)
        r2 = self.conv2[1](x2)
        f2 = self.res_construction[1](f1)
        f2 = r2 + f2
        f2 = torch.cat((torch.sigmoid(f2[:, 0, :, :]).unsqueeze(dim=1), f2[:, 1:, :, :]), dim=1)

        x3 = self.rdbs[2](x2)
        x3 = self.dconv[2](x3)
        r3 = self.conv2[2](x3)
        f3 = self.res_construction[2](f2)
        f3 = r3 + f3
        f3 = torch.cat((torch.sigmoid(f3[:, 0, :, :]).unsqueeze(dim=1), f3[:, 1:, :, :]), dim=1)
        return f3.transpose(0, 1)


# 示例代码
if __name__ == "__main__":
    # 检查是否有 CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResLap(15).to(device)
    a = torch.rand((100, 15, 6, 9)).to(device)
    r = model(a)

    print(r.shape)
