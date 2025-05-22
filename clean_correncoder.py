from tqdm import tqdm
import pandas as pd
import os
import torch
import numpy as np
import random
from sklearn.utils import shuffle
from sklearn.model_selection import KFold

from helperfunctions import remove_flat_subjects
from model2 import Correncoder_model
import torch.nn as nn
from scipy.signal import resample
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.model_selection import LeaveOneGroupOut
from scipy.stats import pearsonr
from helperfunctions import *

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





# a function that gives a rough indication of breaths per minute error by examining the crossings of 0.5
# this assumes that the respiratory reference is normalised between 0 and 1.
def breaths_per_min_zc(output_array_zc, input_array_zc):
    peak_count_output = []
    peak_count_cap = []
    for ind_output in range(output_array_zc.shape[0]):
        output_array_zc_temp = output_array_zc[ind_output, 0, :]
        input_array_zc_temp = input_array_zc[ind_output, :]
        output_array_zc_temp = output_array_zc_temp - 0.5
        input_array_zc_temp = input_array_zc_temp - 0.5
        zero_crossings_output = ((output_array_zc_temp[:-1] * output_array_zc_temp[1:]) < 0).sum()
        zero_crossings_input = ((input_array_zc_temp[:-1] * input_array_zc_temp[1:]) < 0).sum()
        peak_count_output.append(zero_crossings_output)
        peak_count_cap.append(zero_crossings_input)
    peak_count_output = np.array(peak_count_output)
    peak_count_cap = np.array(peak_count_cap)
    # 6.5 is used ot scale up to 1 minute, as each segment here is 60/6.5 seconds long.
    mean_error = ((np.mean(peak_count_output - peak_count_cap)) / 2) * 6.5
    mean_abs_error = ((np.mean(np.abs(peak_count_output - peak_count_cap))) / 2) * 6.5
    return mean_abs_error, mean_error



ppg_csv_path = "/Users/eli/Downloads/PPG Data/csv"
ppg_csv_files = [f for f in os.listdir(ppg_csv_path) if f.endswith('.csv') and not f.startswith('.DS_Store')]
input_name = 'PPG'
target_name = 'NASAL CANULA'

num_of_subjects = 0
original_fs = 256  # your original sampling rate
target_fs = 30  # your desired target rate

cutoff = 1  # Desired cutoff frequency of the filter (Hz)
order = 8  # Order of the filter
# Design Butterworth low-pass filter
nyquist = 0.5 * target_fs  # Nyquist Frequency
normal_cutoff = cutoff / nyquist  # Normalize the frequency
b, a = butter(order, normal_cutoff, btype='low', analog=False)

# define number of epochs and batch size
num_epochs = 50
batch_size = 512

# set a seed for evaluation (optional)
seed_val = 55
print("Seed")
print(seed_val)
torch.manual_seed(seed_val)
random.seed(seed_val)
np.random.seed(seed_val)

# set the learning rate for Adam optimisation
learning_rate = 0.0001

patience = 20
lr_scheduler_patience = 8  # Lower than early stopping patience
# Select device (GPU if available, otherwise CPU)
device = torch.device("mps")
print(f"Using device: {device}")

lowcut = 0.1   # example lower cutoff (Hz), adjust as needed
highcut = 1.1  # example upper cutoff (Hz), adjust as needed
ppg_list = []
resp_list = []
print("data processing...")
for ppg_file in tqdm(ppg_csv_files, leave=True):
    data = pd.read_csv(os.path.join(ppg_csv_path, ppg_file), sep='\t', index_col='Time', skiprows=[1])
    if input_name in data.columns and target_name in data.columns:

        num_of_subjects += 1

        # --- extract both signals ---
        ppg_signal = data[input_name].to_numpy()  # extract the PPG signal
        resp_signal = data[target_name].to_numpy()  # extract the target signal (NASAL CANULA, AIRFLOW, etc.)

        # --- Resample both signals ---
        duration_seconds = ppg_signal.shape[0] / original_fs
        new_length = int(duration_seconds * target_fs)

        ppg_signal = resample(ppg_signal, new_length)
        resp_signal = resample(resp_signal, new_length)

        # --- filter both signals ---
        ppg_signal = apply_bandpass_filter(ppg_signal, lowcut, highcut, target_fs)
        resp_signal = apply_bandpass_filter(resp_signal, lowcut, highcut, target_fs)

        # --- segment both signals ---
        ppg_segments = segment_signal(ppg_signal, segment_length=32 * 30,step_size=242)
        resp_segments = segment_signal(resp_signal, segment_length=32 * 30,step_size=242)

        # --- append the signals to the list ---
        ppg_list.append(ppg_segments)
        resp_list.append(resp_segments)


# --- crop all signals to the same length ---
min_len = min([sig.shape[0] for sig in ppg_list])
ppg_list = [sig[:min_len] for sig in ppg_list]
resp_list = [sig[:min_len] for sig in resp_list]

# --- stack them together ---
data_ppg = np.stack(ppg_list, axis=0)  # shape: (num_subjects, signal_length)
data_resp = np.stack(resp_list, axis=0)  # shape: (num_subjects, signal_length)
print(f"data_ppg shape: {data_ppg.shape}")
print(f"data_resp shape: {data_resp.shape}")


# --- remove flat subjects ---
subject_ids = np.arange(num_of_subjects)
data_ppg, data_resp = remove_flat_subjects(data_ppg, data_resp,subject_ids)
subject_ids = np.arange(data_ppg.shape[0])

# --- check for null in data ---
check_null_or_empty(data_ppg, "data_ppg")
check_null_or_empty(data_resp, "data_resp")

# --- split to train and test sets ---
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed_val)
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
# criterion = torch.nn.MSELoss()
criterion = MSECorrelationLoss(alpha=0.3)  # adjust alpha to your needs
fold_test_losses = []
fold_maes = []
fold_rmses = []
fold_corrs = []

# --- train loop ---
for fold, (train_idx, val_idx) in enumerate(logo.split(data_ppg[train_val_subjects], data_resp[train_val_subjects], groups=train_val_subjects)):
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=lr_scheduler_patience, factor=0.5, verbose=True
    )

    best_val_loss = float('inf')
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        with tqdm(range(0, total_step, batch_size), desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False) as pbar:
            for i in pbar:
                batch_X = trainXT[i:i + batch_size]
                batch_y = trainyT[i:i + batch_size]
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
        print(f"Epoch {epoch + 1}/{num_epochs} Train Loss: {avg_train_loss:.4f} Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model_fold_{fold + 1}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
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
num_samples = 5

# Flatten testX and testy (segments x 1 x segment_length)
testX_flat = testX.reshape(-1, testX.shape[-1])[:, np.newaxis, :]
testy_flat = testy.reshape(-1, testy.shape[-1])

# Randomly choose indices for sampling
random_indices = random.sample(range(testX_flat.shape[0]), num_samples)

# Convert test samples to torch tensors on device
test_samples = torch.from_numpy(testX_flat[random_indices].astype(np.float32)).to(device)

# Load all best models and collect predictions
all_predictions = []
num_folds = 19
for fold in range(1, num_folds + 1):  # assuming folds == train_val_subjects count
    model = Correncoder_model().to(device)
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