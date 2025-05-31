import torch
import torch.nn as nn
import torch.nn.functional as F

# --- DILATED RESIDUAL INCEPTION BLOCK ---
class DRInceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 4, 8]):
        super(DRInceptionBlock, self).__init__()
        # Inception with different dilations
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate)
            for rate in dilation_rates
        ])
        self.bn = nn.BatchNorm1d(out_channels * len(dilation_rates))
        self.res_conv = nn.Conv1d(in_channels, out_channels * len(dilation_rates), kernel_size=1)
        self.res_bn = nn.BatchNorm1d(out_channels * len(dilation_rates))
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # Inception
        outs = [conv(x) for conv in self.convs]
        x_cat = torch.cat(outs, dim=1)
        x_cat = self.bn(x_cat)
        # Residual connection
        res = self.res_conv(x)
        res = self.res_bn(res)
        out = self.activation(x_cat + res)
        return out

# --- ENCODER BLOCK ---
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=4):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=stride, padding=0)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.drinception = DRInceptionBlock(out_channels, out_channels // 2)  # Split channels to keep total similar

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drinception(x)
        return x

# --- DECODER BLOCK ---
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=4)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.drinception = DRInceptionBlock(out_channels, out_channels // 2)

    def forward(self, x, skip):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.act(x)
        # Crop or pad skip connection to match shape
        if x.shape[-1] > skip.shape[-1]:
            x = x[..., :skip.shape[-1]]
        elif x.shape[-1] < skip.shape[-1]:
            skip = skip[..., :x.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        x = self.drinception(x)
        return x

# --- RESP NET MAIN ARCHITECTURE ---
class RespNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(RespNet, self).__init__()
        # Encoder (8 levels)
        filters = [16, 32, 64, 128, 256, 512, 512, 512]  # Per paper
        self.encoders = nn.ModuleList()
        in_ch = input_channels
        for f in filters:
            self.encoders.append(EncoderBlock(in_ch, f))
            in_ch = f * 2  # Due to DRInception block output channel doubling

        # Decoder
        self.decoders = nn.ModuleList()
        rev_filters = list(reversed(filters))
        for i in range(len(rev_filters) - 1):
            # in_channels = previous + skip connection
            self.decoders.append(
                DecoderBlock(rev_filters[i]*2 + rev_filters[i+1]*2, rev_filters[i+1])
            )
        # Final 1x1 conv
        self.final_conv = nn.Conv1d(rev_filters[-1]*2, output_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
        skips = skips[:-1][::-1]
        x = skips.pop(0)  # The last encoder output is the bottleneck
        for dec, skip in zip(self.decoders, skips):
            x = dec(x, skip)
        x = self.final_conv(x)
        return x

# --- LOSS FUNCTION ---
class SmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        return F.smooth_l1_loss(pred, target)

# Example usage:
# model = RespNet(input_channels=1, output_channels=1)
# y_pred = model(torch.randn(16, 1, 2048))  # batch_size=16, 2048 is the input window per paper
# loss = SmoothL1Loss()(y_pred, y_true)
