import torch

import torch.nn as nn
#define number of kernels per layer
n_in, n_out = 1, 8
n_out2 = 8
n_out3 = 8
n_outputs = 1

# define kernel lengths, padding, dilation, stride, and dropout
kernel_size = 75
kernel_size2 = 50
kernel_size3 = 30
padding = 30
dilation = 1
stride = 1
dropout_val = 0.5
padding2 = 20
padding3 = 10
dilation2 = 1
dilation3 = 1
stride2 = 1
stride3 = 1
class Correncoder_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(n_in, n_out, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(n_out, n_out2, kernel_size=kernel_size2, padding=padding2),
            nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(n_out2, n_out3, kernel_size=kernel_size3, padding=padding3),
            nn.Sigmoid(),
            nn.Dropout(dropout_val)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose1d(n_out3, n_out2, kernel_size=kernel_size3, padding=padding3),
            nn.Sigmoid()
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose1d(n_out2, n_out, kernel_size=kernel_size2, padding=padding2),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.ConvTranspose1d(n_out, n_in, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out
