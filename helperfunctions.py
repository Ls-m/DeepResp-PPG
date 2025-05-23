import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import torch
import numpy as np
from scipy.stats import pearsonr


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # data shape: (segments, length) or (length,)
    filtered = filtfilt(b, a, data, axis=-1)
    return filtered


def evaluate_metrics(y_true, y_pred):
    # Convert tensors to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Flatten to 1D arrays if needed
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    rmse = np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2))
    corr, _ = pearsonr(y_true_flat, y_pred_flat)  # Pearson correlation coefficient
    return mae, rmse, corr

def remove_flat_subjects(data_ppg, data_resp, subject_ids):
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
    return data_ppg, data_resp

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


def normalize_data(X, y):
    epsilon = 1e-8
    for i in range(X.shape[0]):
        ppg_range = X[i].max() - X[i].min()
        X[i] = np.zeros_like(X[i]) if ppg_range < epsilon else -1 + 2 * (X[i] - X[i].min()) / (ppg_range + epsilon)
        resp_range = y[i].max() - y[i].min()
        y[i] = np.zeros_like(y[i]) if resp_range < epsilon else (y[i] - y[i].min()) / (resp_range + epsilon)
    return X, y

def check_null_or_empty(arr, arr_name="Array"):
    """
    Check for NaNs, infinite values, or all-zero rows in a NumPy array.

    Args:
        arr (np.ndarray): Input NumPy array.
        arr_name (str): Optional name for the array in printouts.

    Prints a summary if any issues are found.
    """
    nan_mask = np.isnan(arr)
    inf_mask = np.isinf(arr)
    zero_mask = np.all(arr == 0, axis=-1)  # check if entire segment is all zeros

    nan_count = np.sum(nan_mask)
    inf_count = np.sum(inf_mask)
    zero_count = np.sum(zero_mask)

    if nan_count > 0 or inf_count > 0 or zero_count > 0:
        print(f"==== {arr_name} check ====")
        if nan_count > 0:
            print(f"NaN values found: {nan_count}")
        if inf_count > 0:
            print(f"Infinite values found: {inf_count}")
        if zero_count > 0:
            print(f"All-zero segments found: {zero_count} (shape: {arr.shape})")
    else:
        print(f"No NaN, infinite, or all-zero records in {arr_name}.")