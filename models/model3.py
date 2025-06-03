import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

# Based on Section II.A, Equation 4 in respnet.pdf
class SmoothL1Loss(nn.Module):
    """
    Implements the Smooth L1 Loss (Huber Loss) as described in the paper.
    The paper's formula implies a beta (delta) value of 1.0.
    """

    def __init__(self):
        super(SmoothL1Loss, self).__init__()

    def forward(self, y_pred, y_true):
        # Calculate the absolute difference between prediction and ground truth
        diff = torch.abs(y_pred - y_true)

        # Apply the Smooth L1 Loss formula:
        # 0.5 * diff^2 if diff < 1.0
        # diff - 0.5 if diff >= 1.0
        loss = torch.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5)

        # Return the mean loss over the batch
        return torch.mean(loss)


# Based on Section II.B and Figure 2 in respnet.pdf
class DilatedResidualInceptionBlock(nn.Module):
    """
    Implements the Dilated Residual Inception Block used in RespNet.
    This block combines multiple dilated convolutions with different rates
    and a residual connection.
    """

    def __init__(self, in_channels, out_channels):
        super(DilatedResidualInceptionBlock, self).__init__()

        # Ensure input and output channels are the same for the residual connection
        if in_channels != out_channels:
            raise ValueError(
                "For DilatedResidualInceptionBlock, in_channels must be equal to out_channels for residual connection.")

        # The block splits channels into 4 branches, so out_channels must be divisible by 4
        if out_channels % 4 != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by 4 for DilatedResidualInceptionBlock")

        branch_channels = out_channels // 4  # Channels for each inception branch

        # Define the four branches with 1x1 Conv -> BN -> LeakyReLU -> Dilated Conv -> BN -> LeakyReLU
        # Dilated convolutions use kernel_size=4 and 'same' padding to maintain spatial dimensions.
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.LeakyReLU(0.2),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=4, padding='same', dilation=1),  # Dilation rate: 1
            nn.BatchNorm1d(branch_channels),
            nn.LeakyReLU(0.2)
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.LeakyReLU(0.2),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=4, padding='same', dilation=2),  # Dilation rate: 2
            nn.BatchNorm1d(branch_channels),
            nn.LeakyReLU(0.2)
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.LeakyReLU(0.2),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=4, padding='same', dilation=4),  # Dilation rate: 4
            nn.BatchNorm1d(branch_channels),
            nn.LeakyReLU(0.2)
        )
        self.branch4 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.LeakyReLU(0.2),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=4, padding='same', dilation=8),  # Dilation rate: 8
            nn.BatchNorm1d(branch_channels),
            nn.LeakyReLU(0.2)
        )

        # Final 1x1 convolution after concatenating branch outputs, before residual addition
        self.final_conv1x1 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        identity = x  # Store input for the residual connection

        # Process input through each dilated inception branch
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)

        # Concatenate the outputs from all branches along the channel dimension
        concatenated_output = torch.cat([out1, out2, out3, out4], dim=1)

        # Apply final 1x1 convolution, Batch Normalization, and LeakyReLU
        processed_concatenated_output = self.final_relu(self.final_bn(self.final_conv1x1(concatenated_output)))

        # Add the residual connection
        output = identity + processed_concatenated_output
        return output


# Based on Section II.B and Figure 1 in respnet.pdf
class RespNet(nn.Module):
    """
    RespNet is a fully convolutional encoder-decoder network for respiration signal extraction.
    It utilizes Dilated Residual Inception Blocks and skip connections similar to U-Net.
    """

    def __init__(self, input_channels=1, output_channels=1):
        super(RespNet, self).__init__()

        # Define the number of filters for each encoder level.
        # Channels double until 512, then remain 512 for subsequent levels.
        # This creates 8 encoder levels (0-7), with 9 channel values including the input.
        encoder_filters = [input_channels, 32, 64, 128, 256, 512, 512, 512, 512]

        self.encoder_blocks = nn.ModuleList()
        self.inception_blocks_encoder = nn.ModuleList()

        # Build the Encoder path (8 levels)
        for i in range(8):
            in_f = encoder_filters[i]
            out_f = encoder_filters[i + 1]

            # Downsampling strategy:
            # First 5 levels (i=0 to 4) use stride 4 for aggressive downsampling.
            # Last 3 levels (i=5 to 7) use stride 1 with 'same' padding to maintain length.
            stride = 4 if i < 5 else 1
            padding = 'same' if stride == 1 else 0  # 'same' padding for stride 1 to maintain length

            encoder_layer = nn.Sequential(
                nn.Conv1d(in_f, out_f, kernel_size=4, stride=stride, padding=padding),
                nn.BatchNorm1d(out_f),
                nn.LeakyReLU(0.2)
            )
            self.encoder_blocks.append(encoder_layer)

            # Append a Dilated Residual Inception Block after each encoder convolution
            self.inception_blocks_encoder.append(DilatedResidualInceptionBlock(out_f, out_f))

        # Decoder path (8 levels)
        self.decoder_layers = nn.ModuleList()

        # Build the Decoder path
        # The deconvolution operations mirror the encoder's downsampling.
        # First 3 decoder levels (i=0 to 2, corresponding to encoder levels 7, 6, 5) use stride 1.
        # Last 5 decoder levels (i=3 to 7, corresponding to encoder levels 4, 3, 2, 1, 0) use stride 4.
        for i in range(8):
            # Determine input and output channels for the deconvolution layer
            deconv_in_channels = encoder_filters[8 - i]  # Channels from the previous decoder stage or bottleneck
            # Channels after deconvolution, before concatenation. For the last level, it goes to input_channels.
            deconv_out_channels = encoder_filters[7 - i] if i < 7 else input_channels

            # Upsampling stride: 1 for first 3 decoder levels, 4 for the rest
            stride = 1 if i < 3 else 4
            # Padding for ConvTranspose1d.
            # For stride=1, kernel=4, padding=1 gives L_out = L_in + 1.
            # For stride=4, kernel=4, padding=0 gives L_out = (L_in-1)*4 + 4.
            padding = 1 if stride == 1 else 0

            self.decoder_layers.append(
                nn.ConvTranspose1d(deconv_in_channels, deconv_out_channels, kernel_size=4, stride=stride,
                                   padding=padding))

            # Concatenation with skip connection and subsequent processing
            # Skip connection channel for decoder level `i` is from encoder level `7-i`.
            # For the very last decoder level (i=7), the skip connection is the original input `x`.
            skip_channels_for_concat = encoder_filters[7 - i] if i < 7 else input_channels
            concat_channels = deconv_out_channels + skip_channels_for_concat

            # Conditional application of DilatedResidualInceptionBlock:
            # It's applied for the first 7 decoder levels.
            # The very last decoder level (i=7) only has a Conv-BN-ReLU layer after concatenation,
            # as per the paper's description for the final stage.
            if i < 7:  # For the first 7 decoder levels
                self.decoder_layers.append(nn.Sequential(
                    nn.Conv1d(concat_channels, deconv_out_channels, kernel_size=1),
                    # Adjust channels after concatenation
                    nn.BatchNorm1d(deconv_out_channels),
                    nn.LeakyReLU(0.2)
                ))
                self.decoder_layers.append(DilatedResidualInceptionBlock(deconv_out_channels, deconv_out_channels))
            else:  # For the very last decoder level (i=7)
                # Only a Conv-BN-ReLU to combine concatenated features before the final output convolution
                self.decoder_layers.append(nn.Sequential(
                    nn.Conv1d(concat_channels, deconv_out_channels, kernel_size=1),  # Maps e.g., 2 channels to 1
                    nn.BatchNorm1d(deconv_out_channels),
                    nn.LeakyReLU(0.2)
                ))

        # Final 1x1 convolution layer to map the features from the last decoder level
        # to the desired number of output channels (e.g., 1 for respiration signal).
        self.final_output_conv = nn.Conv1d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        # Add a channel dimension if the input is 2D (batch, length)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Becomes (batch, 1, length)

        skip_connections = []  # List to store outputs of encoder inception blocks for skip connections
        x_enc = x  # Initialize encoder input with the original input `x`

        # Encoder path: Process input through each encoder block
        for i in range(len(self.encoder_blocks)):
            x_enc = self.encoder_blocks[i](x_enc)  # Apply convolution, BN, LeakyReLU
            x_enc = self.inception_blocks_encoder[i](x_enc)  # Apply Dilated Residual Inception Block
            skip_connections.append(x_enc)  # Store the output for skip connection

        # Decoder path: Start with the output of the last encoder block (bottleneck)
        x_dec = skip_connections[-1]

        # Iterate through decoder levels
        for i in range(8):
            # Deconvolution layer (first part of the decoder block)
            x_dec = self.decoder_layers[i * 3](x_dec)

            # Get the corresponding skip connection from the encoder
            # For the last decoder level (i=7), the skip connection is the original input `x`.
            current_skip = skip_connections[6 - i] if i < 7 else x

            # Handle spatial dimension mismatch before concatenation
            # If x_dec's length is greater than current_skip's length, crop x_dec.
            # This is necessary because ConvTranspose1d with stride=1, kernel_size=4, padding=1
            # results in an output length of L_in + 1, causing mismatches with the skip connection.
            if x_dec.shape[2] > current_skip.shape[2]:
                diff = x_dec.shape[2] - current_skip.shape[2]
                x_dec = x_dec[:, :, :-diff]  # Crop from the right (end)

            # If x_dec is shorter than current_skip, it would require padding,
            # but based on current layer configurations, x_dec should generally be
            # equal to or longer than current_skip after deconvolution.

            # Concatenate with the corresponding skip connection from the encoder
            x_dec = torch.cat([x_dec, current_skip], dim=1)

            # Apply Conv-BN-ReLU after concatenation (second part of the decoder block)
            x_dec = self.decoder_layers[i * 3 + 1](x_dec)

            # Apply Dilated Residual Inception Block for the first 7 decoder levels
            if i < 7:
                x_dec = self.decoder_layers[i * 3 + 2](x_dec)  # Inception block

        # Apply the final 1x1 convolution to get the desired output
        output = self.final_output_conv(x_dec)
        return output


class RespNetLightning(L.LightningModule):
    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        lr=1e-4,
        loss_type='smoothl1',
        lr_scheduler_factor=0.55,
        lr_scheduler_patience=5
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = RespNet(input_channels=input_channels, output_channels=output_channels)
        if loss_type == 'smoothl1':
            self.criterion = SmoothL1Loss()
        elif loss_type == 'mse':
            self.criterion = nn.MSELoss()
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
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }