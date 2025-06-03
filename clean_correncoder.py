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
from torch.utils.data import DataLoader, Dataset
import lightning as L






class PPGRespDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class PPGRespDataModule(L.LightningDataModule):
    def __init__(self, trainX, trainy, valX, valy, testX, testy, batch_size=128):
        super().__init__()
        self.trainX, self.trainy = trainX, trainy
        self.valX, self.valy = valX, valy
        self.testX, self.testy = testX, testy
        self.batch_size = batch_size
    def setup(self, stage=None):
        self.train_ds = PPGRespDataset(self.trainX, self.trainy)
        self.val_ds = PPGRespDataset(self.valX, self.valy)
        self.test_ds = PPGRespDataset(self.testX, self.testy)
    def train_dataloader(self): return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self): return DataLoader(self.val_ds, batch_size=self.batch_size)
    def test_dataloader(self): return DataLoader(self.test_ds, batch_size=self.batch_size)




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
train_val_ppg = data_ppg[train_val_subjects]
train_val_resp = data_resp[train_val_subjects]
test_ppg = data_ppg[test_subjects]
test_resp = data_resp[test_subjects]




print(f"Train+Val subjects: {len(train_val_subjects)}, Test subjects: {len(test_subjects)}")


# --- normalize test data once and save ---
test_ppg_norm, test_resp_norm = normalize_data(test_ppg.copy(), test_resp.copy())
np.savez('test_data.npz', testX=test_ppg_norm, testy=test_resp_norm)
print("Saved normalized test data.")



logo = LeaveOneGroupOut()
fold_test_losses, fold_maes, fold_rmses, fold_corrs = [], [], [], []
all_predictions = []
train_losses_folds = []
val_losses_folds = []

experiment_name = f"experiment_{time.strftime('%Y%m%d_%H%M%S')}"
# --- train loop ---
for fold, (train_idx, val_idx) in enumerate(logo.split(train_val_ppg, train_val_resp, groups=train_val_subjects)):
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
    testX = test_ppg_norm.reshape(-1, test_ppg_norm.shape[-1])[:, np.newaxis, :]
    testy = test_resp_norm.reshape(-1, test_resp_norm.shape[-1])
    datamodule = PPGRespDataModule(trainX, trainy, valX, valy, testX, testy, batch_size=128)

    model = CorrencoderLightning(alpha=0.9, lr=1e-4)
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    early_stop_callback = L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=20, mode="min")

    # model = Correncoder_model().to(device)
    # model = Transformer1DRegressor(
    #     input_dim=1,
    #     d_model=64,
    #     nhead=4,
    #     num_layers=4,
    #     segment_length=SEGMENT_LENGTH
    # ).to(device)
    # model = RespNet(input_channels=1, output_channels=1).to(device)
    # summary(model, input_size=(1, 1, SEGMENT_LENGTH))
    trainer = L.Trainer(
        max_epochs=20,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",
        devices="auto",
        logger=False,
    )
    trainer.fit(model, datamodule=datamodule)
    # --- Save and plot losses for this fold ---
    train_losses = model.train_loss_history
    val_losses = model.val_loss_history
    # train_losses_folds.append(train_losses)
    # val_losses_folds.append(val_losses)
    plot_loss(train_losses, val_losses)
    test_results = trainer.test(model, datamodule=datamodule, ckpt_path='best')
    # --- Collect all predictions for ensemble ---
    best_model = CorrencoderLightning.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_model.eval()
    preds = []
    for xb, _ in datamodule.test_dataloader():
        xb = xb.to(best_model.device)
        with torch.no_grad():
            preds.append(best_model(xb).cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    all_predictions.append(preds)
    # --- Metrics ---
    y_pred = preds.squeeze(1)
    mae, rmse, corr = evaluate_metrics(testy, y_pred)
    fold_test_losses.append(test_results[0]['test_loss'])
    fold_maes.append(mae)
    fold_rmses.append(rmse)
    fold_corrs.append(corr)
    print(
        f"Fold {fold + 1} Test Loss: {test_results[0]['test_loss']:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, Corr: {corr:.4f}")



        # Log metrics for this fold
    writer.add_scalar("Fold/Test Loss", test_results[0]['test_loss'], 0)
    writer.add_scalar("Fold/MSE", rmse, 0)
    writer.add_scalar("Fold/MAE", mae, 0)
    writer.add_scalar("Fold/Correlation", corr, 0)
    writer.close()



print(f"\nAverage Test Loss: {np.mean(fold_test_losses):.4f} ± {np.std(fold_test_losses):.4f}")
print(f"Average MAE: {np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f}")
print(f"Average RMSE: {np.mean(fold_rmses):.4f} ± {np.std(fold_rmses):.4f}")
print(f"Average Corr: {np.mean(fold_corrs):.4f} ± {np.std(fold_corrs):.4f}")
# After fold_test_losses is computed
np.save('fold_test_losses.npy', np.array(fold_test_losses))
print("Saved fold test losses.")



# ---- 10. Ensemble Prediction and Plot ----
all_predictions = np.stack(all_predictions, axis=0)  # (num_folds, num_samples, 1, segment_length)
ensemble_preds = np.mean(all_predictions, axis=0)    # (num_samples, 1, segment_length)
num_samples_to_plot = 10
random_indices = random.sample(range(ensemble_preds.shape[0]), num_samples_to_plot)
for i, idx in enumerate(random_indices):
    plt.figure(figsize=(12, 3))
    plt.plot(ensemble_preds[idx][0], label='Ensemble Prediction', color='blue')
    plt.plot(testy[idx], label='Ground Truth', color='red', linestyle='--')
    plt.title(f"Test Sample Index {idx}")
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    plt.show()