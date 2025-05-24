import optuna
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

# -- All other helper classes/functions (MSECorrelationLoss, breaths_per_min_zc, etc) are unchanged --
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

ORIGINAL_FS = 256  # your original sampling rate
TARGET_FS = 30  # your desired target rate
LOWCUT = 0.1    # Fixed lowcut
# HIGHCUT is now optimized

SEGMENT_LENGTH = 16 * 30
STEP_SIZE = 242

NUM_EPOCHS = 8           # Make this small for Optuna (faster). Increase later for final train!
BATCH_SIZE = 512
LEARNING_RATE = 1e-4     # Optimized
PATIENCE = 5             # Fewer patience for speed
LR_SCHEDULER_PATIENCE = 3

SEED = 55
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
device = torch.device("mps")

# --- DATA LOADING - Only ONCE outside Optuna ---
ppg_list = []
resp_list = []
num_of_subjects = 0
print("data processing...")
for ppg_file in tqdm(ppg_csv_files, leave=True):
    data = pd.read_csv(os.path.join(DATA_PATH, ppg_file), sep='\t', index_col='Time', skiprows=[1])
    if INPUT_NAME in data.columns and TARGET_NAME in data.columns:
        num_of_subjects += 1
        ppg_signal = data[INPUT_NAME].to_numpy()
        resp_signal = data[TARGET_NAME].to_numpy()
        duration_seconds = ppg_signal.shape[0] / ORIGINAL_FS
        new_length = int(duration_seconds * TARGET_FS)
        ppg_resampled = resample(ppg_signal, new_length)
        resp_resampled = resample(resp_signal, new_length)
        ppg_list.append(ppg_resampled)
        resp_list.append(resp_resampled)

min_len = min([len(ppg) for ppg in ppg_list])
ppg_list = [ppg[:min_len] for ppg in ppg_list]
resp_list = [resp[:min_len] for resp in resp_list]
ppg_mat = np.stack(ppg_list, axis=0)
resp_mat = np.stack(resp_list, axis=0)
subject_ids = np.arange(num_of_subjects)

# --- Split train/val/test ONCE ---
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_val_idx, test_idx = next(gss.split(ppg_mat, groups=subject_ids))
train_val_subjects = subject_ids[train_val_idx]
test_subjects = subject_ids[test_idx]
train_ppg = ppg_mat[train_val_subjects]
train_resp = resp_mat[train_val_subjects]
test_ppg = ppg_mat[test_subjects]
test_resp = resp_mat[test_subjects]

# --- OPTUNA OBJECTIVE FUNCTION ---
def objective(trial):
    # --- Hyperparameters to optimize ---
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    alpha = trial.suggest_uniform('alpha', 0.0, 1.0)
    highcut = trial.suggest_uniform('highcut', 0.5, 2.5)

    # --- Preprocess data using the trial's highcut ---
    ppg_filtered_list, resp_filtered_list = [], []
    for ppg, resp in zip(train_ppg, train_resp):
        ppg_filtered = apply_bandpass_filter(ppg, LOWCUT, highcut, TARGET_FS)
        resp_filtered = apply_bandpass_filter(resp, LOWCUT, highcut, TARGET_FS)
        ppg_filtered_list.append(ppg_filtered)
        resp_filtered_list.append(resp_filtered)
    # Segment
    ppg_segs = [segment_signal(ppg, SEGMENT_LENGTH, STEP_SIZE) for ppg in ppg_filtered_list]
    resp_segs = [segment_signal(resp, SEGMENT_LENGTH, STEP_SIZE) for resp in resp_filtered_list]
    # Crop all signals to the same length
    min_len = min([sig.shape[0] for sig in ppg_segs])
    ppg_segs = [sig[:min_len] for sig in ppg_segs]
    resp_segs = [sig[:min_len] for sig in resp_segs]
    data_ppg = np.stack(ppg_segs, axis=0)
    data_resp = np.stack(resp_segs, axis=0)
    subject_ids = np.arange(data_ppg.shape[0])

    # Use a single fold for tuning speed!
    logo = LeaveOneGroupOut()
    train_idx, val_idx = next(logo.split(data_ppg, data_resp, groups=subject_ids))
    trainX = data_ppg[train_idx]
    trainy = data_resp[train_idx]
    valX = data_ppg[val_idx]
    valy = data_resp[val_idx]
    # Normalize
    trainX, trainy = normalize_data(trainX.copy(), trainy.copy())
    valX, valy = normalize_data(valX.copy(), valy.copy())
    # Flatten and reshape
    trainX = trainX.reshape(-1, trainX.shape[-1])[:, np.newaxis, :]
    trainy = trainy.reshape(-1, trainy.shape[-1])
    valX = valX.reshape(-1, valX.shape[-1])[:, np.newaxis, :]
    valy = valy.reshape(-1, valy.shape[-1])
    # To tensor
    trainXT = torch.from_numpy(trainX.astype(np.float32)).to(device)
    trainyT = torch.from_numpy(trainy.astype(np.float32)).to(device)
    valXT = torch.from_numpy(valX.astype(np.float32)).to(device)
    valyT = torch.from_numpy(valy.astype(np.float32)).to(device)

    # Model, optimizer, loss
    model = Correncoder_model().to(device)
    criterion = MSECorrelationLoss(alpha=alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=LR_SCHEDULER_PATIENCE, factor=0.5, verbose=False
    )

    best_val_loss = float('inf')
    patience_counter = 0
    total_step = trainXT.shape[0]
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        # Just use all data as a batch for speed
        optimizer.zero_grad()
        outputs = model(trainXT)
        loss = criterion(outputs.squeeze(1), trainyT)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_outputs = model(valXT)
            val_loss = criterion(val_outputs.squeeze(1), valyT).item()
        lr_scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    return best_val_loss

# --- RUN OPTUNA STUDY ---
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)  # Increase n_trials for better search

print("Best params found:", study.best_trial.params)
