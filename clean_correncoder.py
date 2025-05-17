from tqdm import tqdm
import pandas as pd
import os
import torch
import numpy as np
import random
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from model2 import Correncoder_model
import torch.nn as nn
from scipy.signal import resample
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.model_selection import LeaveOneGroupOut

def remove_flat(data_ppg, data_resp):
    FLAT_THRESHOLD = 1e-2
    bad_ppg = np.abs(data_ppg.max(axis=-1) - data_ppg.min(axis=-1)) < FLAT_THRESHOLD
    bad_resp = np.abs(data_resp.max(axis=-1) - data_resp.min(axis=-1)) < FLAT_THRESHOLD
    bad_mask = np.logical_or(bad_ppg, bad_resp)
    print(f"Total segments: {data_ppg.size // data_ppg.shape[-1]}")
    print(f"Flat segments: {bad_mask.sum()}")
    data_ppg_clean = []
    data_resp_clean = []
    subject_ids_clean = []
    for subj in range(data_ppg.shape[0]):
        num_flat = bad_mask[subj].sum()
        print(f"Subject {subj}: flat segments = {num_flat}, total segments = {data_ppg.shape[1]}")
        mask = ~bad_mask[subj]
        if mask.sum() > 0:
            data_ppg_clean.append(data_ppg[subj][mask])
            data_resp_clean.append(data_resp[subj][mask])
            subject_ids_clean.extend([subj] * mask.sum())
    data_ppg_clean = np.concatenate(data_ppg_clean, axis=0)
    data_resp_clean = np.concatenate(data_resp_clean, axis=0)
    subject_ids_clean = np.array(subject_ids_clean)
    print(f"Segments after cleaning: {data_ppg_clean.shape[0]}")
    return data_ppg_clean, data_resp_clean, subject_ids_clean

def remove_subject(data_ppg, data_resp, subject_ids):
    # Calculate flat segments per subject
    FLAT_THRESHOLD = 1e-2
    bad_ppg = np.abs(data_ppg.max(axis=-1) - data_ppg.min(axis=-1)) < FLAT_THRESHOLD
    bad_resp = np.abs(data_resp.max(axis=-1) - data_resp.min(axis=-1)) < FLAT_THRESHOLD
    bad_mask = np.logical_or(bad_ppg, bad_resp)

    # Find subjects where all segments are flat
    subjects_to_remove = []
    for subj in range(data_ppg.shape[0]):
        if bad_mask[subj].sum() > 1000:
            print(f"Marking subject {subj} for removal (all segments flat)")
            subjects_to_remove.append(subj)

    subjects_to_keep = [i for i in range(data_ppg.shape[0]) if i not in subjects_to_remove]

    # Remove the identified subjects
    data_ppg = data_ppg[subjects_to_keep]
    data_resp = data_resp[subjects_to_keep]
    subject_ids = subject_ids[subjects_to_keep]  # if you have subject_ids

    print("Subjects removed (all segments flat):", subjects_to_remove)
    print("Shape after subject removal:", data_ppg.shape)
    return data_ppg, data_resp, subject_ids

def segment_signal(signal, segment_length, step_size=None):
    """
    Splits a 1D signal into segments (windows).

    Args:
        signal (1D array or list): The raw signal to segment.
        segment_length (int): Length of each segment.
        step_size (int, optional): Step size between segments (for overlap).
                                   Defaults to segment_length (no overlap).

    Returns:
        segments (2D np.array): Array of shape (num_segments, segment_length)
    """
    if step_size is None:
        step_size = segment_length  # non-overlapping by default

    signal = np.asarray(signal)
    segments = []

    for start in range(0, len(signal) - segment_length + 1, step_size):
        segment = signal[start:start + segment_length]
        segments.append(segment)

    return np.array(segments)

def plot_loss(train_losses, val_losses=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    if val_losses is not None:
        plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

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

epsilon = 1e-8  # small number to avoid division by zero

def normalize_data(trainX, trainy, valX, valy):
    # Normalize train and val data per segment, scaling PPG to [-1, 1] and resp to [0,1]
    for i in range(trainX.shape[0]):
        ppg_range = trainX[i].max() - trainX[i].min()
        trainX[i] = np.zeros_like(trainX[i]) if ppg_range < epsilon else -1 + 2 * (trainX[i] - trainX[i].min()) / (ppg_range + epsilon)
        resp_range = trainy[i].max() - trainy[i].min()
        trainy[i] = np.zeros_like(trainy[i]) if resp_range < epsilon else (trainy[i] - trainy[i].min()) / (resp_range + epsilon)

    for i in range(valX.shape[0]):
        ppg_range = valX[i].max() - valX[i].min()
        valX[i] = np.zeros_like(valX[i]) if ppg_range < epsilon else -1 + 2 * (valX[i] - valX[i].min()) / (ppg_range + epsilon)
        resp_range = valy[i].max() - valy[i].min()
        valy[i] = np.zeros_like(valy[i]) if resp_range < epsilon else (valy[i] - valy[i].min()) / (resp_range + epsilon)

    return trainX, trainy, valX, valy

def normalize_test_data(testX, testy):
    epsilon = 1e-8
    for i in range(testX.shape[0]):
        ppg_range = testX[i].max() - testX[i].min()
        testX[i] = np.zeros_like(testX[i]) if ppg_range < epsilon else -1 + 2 * (testX[i] - testX[i].min()) / (ppg_range + epsilon)
        resp_range = testy[i].max() - testy[i].min()
        testy[i] = np.zeros_like(testy[i]) if resp_range < epsilon else (testy[i] - testy[i].min()) / (resp_range + epsilon)
    return testX, testy

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
batch_size = 256

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

# Select device (GPU if available, otherwise CPU)
device = torch.device("mps")
print(f"Using device: {device}")


ppg_list = []
resp_list = []
print("data processing...")
for ppg_file in tqdm(ppg_csv_files, leave=True):
    data = pd.read_csv(os.path.join(ppg_csv_path, ppg_file), sep='\t', index_col='Time', skiprows=[1])
    if input_name in data.columns and target_name in data.columns:
        num_of_subjects += 1
        ppg_signal = data[input_name].to_numpy()  # extract the PPG signal
        resp_signal = data[target_name].to_numpy()  # extract the target signal (NASAL CANULA, AIRFLOW, etc.)

        # --- Resample both signals ---
        duration_seconds = ppg_signal.shape[0] / original_fs
        new_length = int(duration_seconds * target_fs)

        ppg_signal = resample(ppg_signal, new_length)
        resp_signal = resample(resp_signal, new_length)
        # print(new_length)
        # orig_indices = np.arange(new_length) * (256 - 1) / (30 - 1)

        # To get the closest integer indices:
        # closest_indices = np.round(orig_indices).astype(int)
        # print(closest_indices[7*480:8*480])  # integer indices in the original array
        #
        # print(ppg_file," ",closest_indices)
        # resp_signal = filtfilt(b, a, resp_signal)

        # Segment the signals into non-overlapping windows (use segment length as 16*30)
        ppg_segments = segment_signal(ppg_signal, segment_length=16 * 30)
        resp_segments = segment_signal(resp_signal, segment_length=16 * 30)

        # Append the segmented signals to the list
        ppg_list.append(ppg_segments)
        resp_list.append(resp_segments)

# First, find the minimum length
min_len = min([sig.shape[0] for sig in ppg_list])

# Then crop all signals to the same length
ppg_list = [sig[:min_len] for sig in ppg_list]
resp_list = [sig[:min_len] for sig in resp_list]
# After looping over all subjects, stack them together
data_ppg = np.stack(ppg_list, axis=0)  # shape: (num_subjects, signal_length)
data_resp = np.stack(resp_list, axis=0)  # shape: (num_subjects, signal_length)

print(f"data_ppg shape: {data_ppg.shape}")
print(f"data_resp shape: {data_resp.shape}")


remove_flat(data_ppg, data_resp)
subject_ids = np.arange(num_of_subjects)
data_ppg, data_resp, subject_ids = remove_subject(data_ppg, data_resp,subject_ids)
subject_ids = np.arange(data_ppg.shape[0])
# Step 2: Split subjects into train_val and test sets (e.g. 80% train_val, 20% test)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed_val)
train_val_idx, test_idx = next(gss.split(data_ppg, groups=subject_ids))

train_val_subjects = subject_ids[train_val_idx]
test_subjects = subject_ids[test_idx]

# Extract test data (kept completely separate)
testX = data_ppg[test_subjects]
testy = data_resp[test_subjects]

print(f"Train+Val subjects: {len(train_val_subjects)}, Test subjects: {len(test_subjects)}")



sub_num = 1

if np.any(np.isnan(data_ppg)):
    print(f"NaNs found in data_ppg")

if np.any(np.isnan(data_resp)):
    print(f"NaNs found in data_resp")




fold_test_losses = []
logo = LeaveOneGroupOut()
criterion = torch.nn.MSELoss()


# Identify flat PPG or resp segments
# bad_ppg = np.abs(testX.max(axis=-1) - testX.min(axis=-1)) < 1e-2
# bad_resp = np.abs(testy.max(axis=-1) - testy.min(axis=-1)) < 1e-2
# bad_mask = np.logical_or(bad_ppg, bad_resp)
# print("Number of flat segments:", bad_mask.sum())
#
# indices = np.argwhere(bad_mask)
# for i in range(min(5, len(indices))):
#     s, seg = indices[i]
#     plt.figure()
#     plt.plot(testX[s, seg], label='PPG')
#     plt.plot(testy[s, seg], label='Resp')
#     plt.title(f"Nearly-Flat Segment subj={s} seg={seg}")
#     plt.legend()
#     plt.show()
# # --- Visualize raw signals before normalization ---
#
# num_subjects = testX.shape[0]
# num_segments = testX.shape[1]
#
# # Pick 3 random subject-segment pairs to visualize
# np.random.seed(42)
# pairs = []
# while len(pairs) < 3:
#     s = np.random.randint(0, num_subjects)
#     seg = np.random.randint(0, num_segments)
#     if (s, seg) not in pairs:
#         pairs.append((s, seg))
#
# plt.figure(figsize=(16, 5))
# for i, (s, seg) in enumerate(pairs):
#     plt.subplot(2, 3, i+1)
#     plt.plot(testX[s, seg], label='Raw PPG')
#     plt.plot(testy[s, seg], label='Raw Resp')
#     plt.title(f'Raw: subj={s} seg={seg}')
#     plt.legend()
# plt.suptitle('Raw PPG & Respiration segments before normalization')
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()
#
# # --- Now normalize and plot the same segments after normalization ---
# testX_norm, testy_norm = normalize_test_data(testX.copy(), testy.copy())
#
# plt.figure(figsize=(16, 5))
# for i, (s, seg) in enumerate(pairs):
#     plt.subplot(2, 3, i+1)
#     plt.plot(testX_norm[s, seg], label='Norm PPG')
#     plt.plot(testy_norm[s, seg], label='Norm Resp')
#     plt.title(f'Norm: subj={s} seg={seg}')
#     plt.legend()
# plt.suptitle('PPG & Respiration segments after normalization')
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()


# exit()



# Normalize test data once and save
testX_norm, testy_norm = normalize_test_data(testX.copy(), testy.copy())
np.savez('test_data.npz', testX=testX_norm, testy=testy_norm)
print("Saved normalized test data.")

for fold, (train_idx, val_idx) in enumerate(logo.split(data_ppg[train_val_subjects], data_resp[train_val_subjects], groups=train_val_subjects)):
    print(f"Fold {fold + 1}/{len(train_val_subjects)}")

    train_subjects = train_val_subjects[train_idx]
    val_subject = train_val_subjects[val_idx]

    trainX = data_ppg[train_subjects]
    trainy = data_resp[train_subjects]
    valX = data_ppg[val_subject]
    valy = data_resp[val_subject]

    # Normalize
    trainX, trainy, valX, valy = normalize_data(trainX.copy(), trainy.copy(), valX.copy(), valy.copy())

    # Flatten and reshape
    trainX = trainX.reshape(-1, trainX.shape[-1])[:, np.newaxis, :]
    trainy = trainy.reshape(-1, trainy.shape[-1])
    valX = valX.reshape(-1, valX.shape[-1])[:, np.newaxis, :]
    valy = valy.reshape(-1, valy.shape[-1])

    trainXT = torch.from_numpy(trainX.astype(np.float32)).to(device)
    trainyT = torch.from_numpy(trainy.astype(np.float32)).to(device)
    valXT = torch.from_numpy(valX.astype(np.float32)).to(device)
    valyT = torch.from_numpy(valy.astype(np.float32)).to(device)

    total_step = trainXT.shape[0]

    model = Correncoder_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

    print(f"Fold {fold + 1} Test Loss: {test_loss:.4f}")
    fold_test_losses.append(test_loss)

print(f"Average Test Loss over all folds: {np.mean(fold_test_losses):.4f}")

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

for fold in range(1, len(train_val_subjects) + 1):  # assuming folds == train_val_subjects count
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