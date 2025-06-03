import torch
import lightning as L

import torch.nn as nn


class MSECorrelationLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(MSECorrelationLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha  # weight for correlation loss

    def forward(self, y_pred, y_true):
        mse_loss = self.mse(y_pred, y_true)

        # Flatten to (batch, -1) for segment-wise correlation if needed
        y_pred_flat = y_pred.view(y_pred.size(0), -1)
        y_true_flat = y_true.view(y_true.size(0), -1)

        # Calculate mean
        mean_pred = torch.mean(y_pred_flat, dim=1, keepdim=True)
        mean_true = torch.mean(y_true_flat, dim=1, keepdim=True)

        # Subtract mean
        y_pred_centered = y_pred_flat - mean_pred
        y_true_centered = y_true_flat - mean_true

        # Compute correlation per batch and average
        numerator = torch.sum(y_pred_centered * y_true_centered, dim=1)
        denominator = torch.sqrt(torch.sum(y_pred_centered ** 2, dim=1) * torch.sum(y_true_centered ** 2, dim=1) + 1e-8)
        corr = numerator / denominator

        # The correlation loss (maximize corr -> minimize (1 - corr))
        corr_loss = 1 - torch.mean(corr)

        # Combine losses
        total_loss = mse_loss + self.alpha * corr_loss
        return total_loss
# ---- 4. Model ----
class CorrencoderLightning(L.LightningModule):
    def __init__(self, alpha=0.3, lr=1e-4, lr_scheduler_factor=0.55, lr_scheduler_patience=3):
        super().__init__()
        n_in, n_out = 1, 8
        n_out2 = 8
        n_out3 = 8
        kernel_size = 75
        kernel_size2 = 50
        kernel_size3 = 30
        padding = 30
        padding2 = 20
        padding3 = 10
        dropout_val = 0.1

        self.save_hyperparameters()
        self.train_loss_history = []
        self.val_loss_history = []
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.layer1 = nn.Sequential(
            nn.Conv1d(n_in, n_out, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(n_out),
            nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(n_out, n_out2, kernel_size=kernel_size2, padding=padding2),
            nn.BatchNorm1d(n_out2),
            nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(n_out2, n_out3, kernel_size=kernel_size3, padding=padding3),
            nn.BatchNorm1d(n_out3),
            nn.Sigmoid(),
            nn.Dropout(dropout_val)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose1d(n_out3, n_out2, kernel_size=kernel_size3, padding=padding3),
            nn.BatchNorm1d(n_out2),
            nn.Sigmoid()
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose1d(n_out2, n_out, kernel_size=kernel_size2, padding=padding2),
            nn.BatchNorm1d(n_out),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.ConvTranspose1d(n_out, n_in, kernel_size=kernel_size, padding=padding)
        )
        self.criterion = MSECorrelationLoss(alpha=alpha)
        self.lr = lr

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(1), y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        # This runs at the end of every train epoch
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
        # This runs at the end of every validation epoch
        epoch_loss = self.trainer.callback_metrics.get('val_loss')
        if epoch_loss is not None:
            self.val_loss_history.append(float(epoch_loss.cpu()))

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(1), y)
        self.log('test_loss', loss, on_epoch=True)
        # For per-batch metrics
        return {'y_true': y.cpu().numpy(), 'y_pred': y_hat.squeeze(1).cpu().numpy()}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=self.lr_scheduler_patience, factor=self.lr_scheduler_factor, verbose=False)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }