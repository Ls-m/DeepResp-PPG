import torch
import torch.nn as nn
from models.model2 import MSECorrelationLoss
import lightning as L

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


class Transformer1DLightning(L.LightningModule):
    def __init__(
        self,
        input_dim=1,
        d_model=64,
        nhead=4,
        num_layers=4,
        segment_length=2048,
        lr=1e-4,
        loss_type='mse_corr',    # <--- default to your custom loss
        alpha=0.5,               # <--- add alpha!
        lr_scheduler_factor=0.55,
        lr_scheduler_patience=5
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = Transformer1DRegressor(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            segment_length=segment_length
        )
        # Choose loss
        if loss_type == 'mse_corr':
            self.criterion = MSECorrelationLoss(alpha=alpha)
        elif loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'smoothl1':
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        self.lr = lr
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.train_loss_history = []
        self.val_loss_history = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(1), y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        epoch_loss = self.trainer.callback_metrics.get('train_loss')
        if epoch_loss is not None:
            self.train_loss_history.append(float(epoch_loss.cpu()))

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(1), y)
        self.log('val_loss', loss, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        epoch_loss = self.trainer.callback_metrics.get('val_loss')
        if epoch_loss is not None:
            self.val_loss_history.append(float(epoch_loss.cpu()))

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(1), y)
        self.log('test_loss', loss, on_epoch=True)
        return {'y_true': y.cpu().numpy(), 'y_pred': y_hat.squeeze(1).cpu().numpy()}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=self.lr_scheduler_patience,
            factor=self.lr_scheduler_factor,

        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }