import torch
import torch.nn as nn

class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(2)].permute(0, 2, 1)

class Transformer1DRegressor(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=4, segment_length=2048):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, d_model, kernel_size=1)
        self.pos_enc = PositionalEncoding1D(d_model, max_len=segment_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Conv1d(d_model, 1, kernel_size=1)
        self.segment_length = segment_length

    def forward(self, x):
        # x: (B, 1, L)
        out = self.input_proj(x)  # (B, d_model, L)
        out = self.pos_enc(out)
        out = out.permute(0, 2, 1)  # (B, L, d_model)
        out = self.transformer(out)
        out = out.permute(0, 2, 1)  # (B, d_model, L)
        out = self.output_proj(out)  # (B, 1, L)
        return out
