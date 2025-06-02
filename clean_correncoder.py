from tqdm import tqdm
import os
import random
from torch.utils.tensorboard import SummaryWriter
import time
from models.model2 import *
from models.model3 import RespNet
from models.model4 import *
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import LeaveOneGroupOut
from helperfunctions import *
import scipy
import pandas as pd
from scipy.signal import resample




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







# --- Configurations ---
DATA_PATH = "/Users/eli/Downloads/PPG Data/csv"
INPUT_NAME = 'PPG'
TARGET_NAME = 'NASAL CANULA'
ppg_csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv') and not f.startswith('.DS_Store')]

BIDMC_PATH = '/Users/eli/Downloads/bidmc_data.mat'

ORIGINAL_FS = 256  # your original sampling rate
TARGET_FS = 30  # your desired target rate
LOWCUT = 0.1
HIGHCUT = 1.3

SEGMENT_LENGTH = 8*30
STEP_SIZE = 242

# criterion = torch.nn.MSELoss()
criterion = MSECorrelationLoss(alpha=0.9)  # adjust alpha to your needs
# criterion = SmoothL1Loss()
NUM_EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
PATIENCE = 20
LR_SCHEDULER_PATIENCE = 8  # Lower than early stopping patience
LR_SCHEDULER_FACTOR = 0.55

SEED = 55
print("Seed")
print(SEED)
# --- Set seeds ---
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


device = torch.device("mps")
print(f"Using device: {device}")





# --- pre-process data ---
# data_ppg, data_resp = preprocess_dataset(kind=0,
#                       ppg_csv_files=ppg_csv_files,
#                       DATA_PATH=DATA_PATH,
#                       INPUT_NAME=INPUT_NAME,
#                       TARGET_NAME=TARGET_NAME,
#                       ORIGINAL_FS=ORIGINAL_FS,
#                       TARGET_FS=TARGET_FS,
#                       LOWCUT=LOWCUT,
#                       HIGHCUT=HIGHCUT,
#                       SEGMENT_LENGTH=SEGMENT_LENGTH,
#                       STEP_SIZE=STEP_SIZE,
#                       bidmc_mat_path=BIDMC_PATH)


# # --- save pre-processed data ---
# np.save('data_ppg.npy', data_ppg)
# np.save('data_resp.npy', data_resp)


# --- load pre-processed data ---
data_ppg = np.load('data_ppg.npy')
data_resp = np.load('data_resp.npy')
num_of_subjects = data_ppg.shape[0]

# --- remove flat subjects ---
subject_ids = np.arange(num_of_subjects)
data_ppg, data_resp = remove_flat_subjects(data_ppg, data_resp,subject_ids)
subject_ids = np.arange(data_ppg.shape[0])

# --- check for null in data ---
check_null_or_empty(data_ppg, "data_ppg")
check_null_or_empty(data_resp, "data_resp")

# --- split to train and test sets ---
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_val_idx, test_idx = next(gss.split(data_ppg, groups=subject_ids))

train_val_subjects = subject_ids[train_val_idx]
test_subjects = subject_ids[test_idx]


testX = data_ppg[test_subjects]
testy = data_resp[test_subjects]

print(f"Train+Val subjects: {len(train_val_subjects)}, Test subjects: {len(test_subjects)}")


# --- normalize test data once and save ---
testX_norm, testy_norm = normalize_data(testX.copy(), testy.copy())
np.savez('test_data.npz', testX=testX_norm, testy=testy_norm)
print("Saved normalized test data.")



logo = LeaveOneGroupOut()


fold_test_losses = []
fold_maes = []
fold_rmses = []
fold_corrs = []


experiment_name = f"experiment_{time.strftime('%Y%m%d_%H%M%S')}"
# --- train loop ---
for fold, (train_idx, val_idx) in enumerate(logo.split(data_ppg[train_val_subjects], data_resp[train_val_subjects], groups=train_val_subjects)):
    writer = SummaryWriter(log_dir=f"runs/{experiment_name}/fold_{fold + 1}")
    print(f"Fold {fold + 1}/{len(train_val_subjects)}")

    # --- split to train and validation sets ---
    train_subjects = train_val_subjects[train_idx]
    val_subject = train_val_subjects[val_idx]

    trainX = data_ppg[train_subjects]
    trainy = data_resp[train_subjects]
    valX = data_ppg[val_subject]
    valy = data_resp[val_subject]

    # --- normalize train and validation ---
    trainX, trainy = normalize_data(trainX.copy(), trainy.copy())
    valX, valy = normalize_data(valX.copy(), valy.copy())

    # --- flatten and reshape ---
    trainX = trainX.reshape(-1, trainX.shape[-1])[:, np.newaxis, :]
    trainy = trainy.reshape(-1, trainy.shape[-1])
    valX = valX.reshape(-1, valX.shape[-1])[:, np.newaxis, :]
    valy = valy.reshape(-1, valy.shape[-1])

    # --- to tensor ---
    trainXT = torch.from_numpy(trainX.astype(np.float32)).to(device)
    trainyT = torch.from_numpy(trainy.astype(np.float32)).to(device)
    valXT = torch.from_numpy(valX.astype(np.float32)).to(device)
    valyT = torch.from_numpy(valy.astype(np.float32)).to(device)

    total_step = trainXT.shape[0]

    model = Correncoder_model().to(device)
    # model = Transformer1DRegressor(
    #     input_dim=1,
    #     d_model=64,
    #     nhead=4,
    #     num_layers=4,
    #     segment_length=SEGMENT_LENGTH
    # ).to(device)
    # model = RespNet(input_channels=1, output_channels=1).to(device)
    # summary(model, input_size=(1, 1, SEGMENT_LENGTH))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=LR_SCHEDULER_PATIENCE, factor=LR_SCHEDULER_FACTOR, verbose=True
    )

    best_val_loss = float('inf')
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0

        with tqdm(range(0, total_step, BATCH_SIZE), desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", leave=False) as pbar:
            for i in pbar:
                batch_X = trainXT[i:i + BATCH_SIZE]
                batch_y = trainyT[i:i + BATCH_SIZE]
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(1), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
                pbar.set_postfix({'Batch Loss': loss.item()})

        avg_train_loss = epoch_loss / total_step
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_outputs = model(valXT)
            val_loss = criterion(val_outputs.squeeze(1), valyT).item()
        val_losses.append(val_loss)

        # Step the LR scheduler
        lr_scheduler.step(val_loss)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} Train Loss: {avg_train_loss:.4f} Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model_fold_{fold + 1}.pth")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    plot_loss(train_losses, val_losses)

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(f"best_model_fold_{fold + 1}.pth"))
    model.eval()

    # Normalize test data with the same function
    # testX_norm, testy_norm = normalize_test_data(testX.copy(), testy.copy())


    testX_flat = testX_norm.reshape(-1, testX_norm.shape[-1])[:, np.newaxis, :]
    testy_flat = testy_norm.reshape(-1, testy_norm.shape[-1])

    testXT = torch.from_numpy(testX_flat.astype(np.float32)).to(device)
    testyT = torch.from_numpy(testy_flat.astype(np.float32)).to(device)

    with torch.no_grad():
        test_outputs = model(testXT)
        test_loss = criterion(test_outputs.squeeze(1), testyT).item()

        # Compute metrics
        mae, rmse, corr = evaluate_metrics(testyT, test_outputs.squeeze(1))
        print(f"Correlation for this fold is: {corr:.4f}")
        # Log metrics for this fold
    writer.add_scalar("Fold/Test Loss", test_loss, 0)
    writer.add_scalar("Fold/MSE", rmse, 0)
    writer.add_scalar("Fold/MAE", mae, 0)
    writer.add_scalar("Fold/Correlation", corr, 0)
    writer.add_scalar("Fold/Num Epochs", epoch + 1, 0)  # number of epochs actually run
    writer.close()
    print(f"Fold {fold + 1} Test Loss: {test_loss:.4f}")
    fold_test_losses.append(test_loss)
    fold_maes.append(mae)
    fold_rmses.append(rmse)
    fold_corrs.append(corr)

print(f"Average Test Loss over all folds: {np.mean(fold_test_losses):.4f}")
print(f"Average MAE over folds: {np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f}")
print(f"Average RMSE over folds: {np.mean(fold_rmses):.4f} ± {np.std(fold_rmses):.4f}")
print(f"Average Correlation over folds: {np.mean(fold_corrs):.4f} ± {np.std(fold_corrs):.4f}")
# After fold_test_losses is computed
np.save('fold_test_losses.npy', np.array(fold_test_losses))
print("Saved fold test losses.")



# Number of random samples to pick
num_samples = 10

# Flatten testX and testy (segments x 1 x segment_length)
testX_flat = testX_norm.reshape(-1, testX_norm.shape[-1])[:, np.newaxis, :]
testy_flat = testy_norm.reshape(-1, testy_norm.shape[-1])

# Randomly choose indices for sampling
random_indices = random.sample(range(testX_flat.shape[0]), num_samples)
print("random indices: ", random_indices)
# Convert test samples to torch tensors on device
test_samples = torch.from_numpy(testX_flat[random_indices].astype(np.float32)).to(device)

# Load all best models and collect predictions
all_predictions = []
num_folds = 19
for fold in range(1, num_folds + 1):  # assuming folds == train_val_subjects count
    model = Correncoder_model().to(device)
    # model = RespNet(input_channels=1, output_channels=1).to(device)
    model.load_state_dict(torch.load(f"best_model_fold_{fold}.pth"))
    model.eval()
    with torch.no_grad():
        preds = model(test_samples).cpu().numpy()  # shape: (num_samples, 1, segment_length)
        all_predictions.append(preds)

# Average predictions across models (folds)
ensemble_preds = np.mean(all_predictions, axis=0)  # shape: (num_samples, 1, segment_length)

# Plot predictions vs ground truth
for i, idx in enumerate(random_indices):
    plt.figure(figsize=(12, 3))
    plt.plot(ensemble_preds[i][0], label='Ensemble Prediction', color='blue')
    plt.plot(testy_flat[idx], label='Ground Truth', color='red', linestyle='--')
    plt.title(f"Test Sample Index {idx}")
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    plt.show()