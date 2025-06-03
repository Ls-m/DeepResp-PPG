import optuna
import os
import torch
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from scipy.signal import resample
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from helperfunctions import apply_bandpass_filter, segment_signal, normalize_data
from models.model2 import CorrencoderLightning  # make sure this is your Lightning model!
from models.model3 import RespNetLightning  # or the actual path to your RespNetLightning class
from models.model4 import *
# Set your config
DATA_PATH = "/Users/eli/Downloads/PPG Data/csv"
INPUT_NAME = 'PPG'
TARGET_NAME = 'NASAL CANULA'
ppg_csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv') and not f.startswith('.DS_Store')]
ORIGINAL_FS = 256
TARGET_FS = 30
LOWCUT = 0.1
STEP_SIZE = 242
NUM_EPOCHS = 8
PATIENCE = 5
LR_SCHEDULER_PATIENCE = 3
SEED = 55

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ------------- Dataset and DataModule -------------
class PPGRespDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class PPGRespDataModule(L.LightningDataModule):
    def __init__(self, trainX, trainy, valX, valy, batch_size=128):
        super().__init__()
        self.trainX, self.trainy = trainX, trainy
        self.valX, self.valy = valX, valy
        self.batch_size = batch_size
    def setup(self, stage=None):
        self.train_ds = PPGRespDataset(self.trainX, self.trainy)
        self.val_ds = PPGRespDataset(self.valX, self.valy)
    def train_dataloader(self): return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self): return DataLoader(self.val_ds, batch_size=self.batch_size)

# ------------- Pre-load all subject data once -------------
ppg_list, resp_list = [], []
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

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_val_idx, test_idx = next(gss.split(ppg_mat, groups=subject_ids))
train_val_subjects = subject_ids[train_val_idx]
train_ppg = ppg_mat[train_val_subjects]
train_resp = resp_mat[train_val_subjects]

# ------------- OPTUNA OBJECTIVE -------------
def objective(trial):
    # --- Hyperparameters to search ---
    # alpha = trial.suggest_float('alpha', 0.3, 1.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    highcut = trial.suggest_float('highcut', 0.5, 2.5)
    lr_scheduler_factor = trial.suggest_float('lr_scheduler_factor', 0.1, 0.9)
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
    # segment_length = trial.suggest_categorical('segment_length', [8*30, 16*30, 32*30, 64*30])
    segment_length = 2048

    # --- Data preprocessing using trial params ---
    ppg_filtered_list, resp_filtered_list = [], []
    for ppg, resp in zip(train_ppg, train_resp):
        ppg_filtered = apply_bandpass_filter(ppg, LOWCUT, highcut, TARGET_FS)
        resp_filtered = apply_bandpass_filter(resp, LOWCUT, highcut, TARGET_FS)
        ppg_filtered_list.append(ppg_filtered)
        resp_filtered_list.append(resp_filtered)
    ppg_segs = [segment_signal(ppg, segment_length, STEP_SIZE) for ppg in ppg_filtered_list]
    resp_segs = [segment_signal(resp, segment_length, STEP_SIZE) for resp in resp_filtered_list]
    min_len = min([sig.shape[0] for sig in ppg_segs])
    ppg_segs = [sig[:min_len] for sig in ppg_segs]
    resp_segs = [sig[:min_len] for sig in resp_segs]
    data_ppg = np.stack(ppg_segs, axis=0)
    data_resp = np.stack(resp_segs, axis=0)
    subject_ids = np.arange(data_ppg.shape[0])

    # --- Grouped train/val split (LOGO) ---
    logo = LeaveOneGroupOut()
    train_idx, val_idx = next(logo.split(data_ppg, data_resp, groups=subject_ids))
    trainX = data_ppg[train_idx]
    trainy = data_resp[train_idx]
    valX = data_ppg[val_idx]
    valy = data_resp[val_idx]

    # --- Normalize ---
    trainX, trainy = normalize_data(trainX.copy(), trainy.copy())
    valX, valy = normalize_data(valX.copy(), valy.copy())
    trainX = trainX.reshape(-1, trainX.shape[-1])[:, np.newaxis, :]
    trainy = trainy.reshape(-1, trainy.shape[-1])
    valX = valX.reshape(-1, valX.shape[-1])[:, np.newaxis, :]
    valy = valy.reshape(-1, valy.shape[-1])

    # --- Lightning DataModule ---
    datamodule = PPGRespDataModule(trainX, trainy, valX, valy, batch_size=batch_size)

    # --- Lightning Model ---
    # model = CorrencoderLightning(
    #     alpha=1, lr=learning_rate,
    #     lr_scheduler_factor=lr_scheduler_factor,
    #     lr_scheduler_patience=LR_SCHEDULER_PATIENCE
    # )
    # model = RespNetLightning(
    #     lr=learning_rate,
    #     lr_scheduler_factor=lr_scheduler_factor,
    #     lr_scheduler_patience=LR_SCHEDULER_PATIENCE,  # from your config
    #     # optionally: loss_type=trial.suggest_categorical('loss_type', ['smoothl1', 'mse'])
    # )
    model = Transformer1DLightning(
        input_dim=1,
        d_model=64,
        nhead=4,
        num_layers=4,
        segment_length=segment_length,  # from trial
        lr=learning_rate,
        loss_type='mse_corr',
        alpha=1,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_patience=LR_SCHEDULER_PATIENCE,
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")
    checkpoint = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")

    # --- Lightning Trainer ---
    trainer = Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="auto",
        devices="auto",
        callbacks=[early_stop, checkpoint],
        logger=False,
        enable_progress_bar=False,
    )

    trainer.fit(model, datamodule=datamodule)
    best_val_loss = trainer.callback_metrics["val_loss"].item()
    return best_val_loss

# ------------- Run Optuna -------------
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1)  # set your number of trials

print("Best params found:", study.best_trial.params)
