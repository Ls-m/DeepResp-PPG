import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import torch
import numpy as np
from scipy.stats import pearsonr


def plot_preprocessing(original, resampled, filtered, fs_orig, fs_target, file_name, label='PPG', seconds=20):
    """
    Plot original, resampled, and filtered signals (zoomed on first N seconds).
    """
    plt.figure(figsize=(16, 4))
    N = int(seconds * fs_orig)
    N_resamp = int(seconds * fs_target)

    # Original
    plt.subplot(1, 3, 1)
    plt.plot(np.arange(N) / fs_orig, original[:N], color='gray')
    plt.title('Original')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Resampled
    plt.subplot(1, 3, 2)
    plt.plot(np.arange(N_resamp) / fs_target, resampled[:N_resamp], color='orange')
    plt.title('Resampled')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Filtered
    plt.subplot(1, 3, 3)
    plt.plot(np.arange(N_resamp) / fs_target, filtered[:N_resamp], color='blue')
    plt.title('Filtered (Resampled + Bandpass)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.suptitle(f"{label} | {file_name}")
    plt.tight_layout()
    plt.show()
def plot_fft(signal, fs, label, filename=None, xlim=None):
    """Plot the FFT of the signal up to xlim Hz (default: Nyquist)."""
    N = len(signal)
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, d=1/fs)
    mask = freqs >= 0
    plt.figure(figsize=(8, 4))
    plt.plot(freqs[mask], np.abs(fft)[mask])
    plt.title(f"FFT of {label}" + (f" | {filename}" if filename else ""))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.ylim(0, 9e5)
    if xlim:
        plt.xlim([0, xlim])
    plt.tight_layout()
    plt.show()

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



def process_signals(ppg_signal, resp_signal,
                   ORIGINAL_FS, TARGET_FS,
                   LOWCUT, HIGHCUT,
                   SEGMENT_LENGTH, STEP_SIZE
                   ):

    # --- Resample both signals ---
    duration_seconds = ppg_signal.shape[0] / ORIGINAL_FS
    new_length = int(duration_seconds * TARGET_FS)
    ppg_resampled = resample(ppg_signal, new_length)
    resp_resampled = resample(resp_signal, new_length)


    # --- Skip if too short for filtering ---
    if len(ppg_resampled) <= 33 or len(resp_resampled) <= 33:
        raise ValueError("Signal too short for filtering")
    # --- Filter both signals ---
    ppg_filtered = apply_bandpass_filter(ppg_resampled, LOWCUT, HIGHCUT, TARGET_FS)
    resp_filtered = apply_bandpass_filter(resp_resampled, LOWCUT, HIGHCUT, TARGET_FS)

    # --- Segment both signals ---
    ppg_segments = segment_signal(ppg_filtered, segment_length=SEGMENT_LENGTH, step_size=STEP_SIZE)
    resp_segments = segment_signal(resp_filtered, segment_length=SEGMENT_LENGTH, step_size=STEP_SIZE)

    return ppg_resampled,ppg_filtered,ppg_segments, resp_segments

def preprocess_dataset(kind=0,
                      ppg_csv_files=None,
                      DATA_PATH=None,
                      INPUT_NAME=None,
                      TARGET_NAME=None,
                      ORIGINAL_FS=None,
                      TARGET_FS=None,
                      LOWCUT=None,
                      HIGHCUT=None,
                      SEGMENT_LENGTH=None,
                      STEP_SIZE=None,
                      bidmc_mat_path=None):
    ppg_list = []
    resp_list = []
    num_of_subjects = 0

    if kind == 0:
        print("Processing your own dataset...")
        for ppg_file in tqdm(ppg_csv_files, leave=True):
            data = pd.read_csv(os.path.join(DATA_PATH, ppg_file), sep='\t', index_col='Time', skiprows=[1])
            if INPUT_NAME in data.columns and TARGET_NAME in data.columns:
                num_of_subjects += 1
                ppg_signal = data[INPUT_NAME].to_numpy()
                resp_signal = data[TARGET_NAME].to_numpy()
                ppg_resampled,ppg_filtered,ppg_segments, resp_segments = process_signals(
                    ppg_signal, resp_signal,
                    ORIGINAL_FS, TARGET_FS,
                    LOWCUT, HIGHCUT,
                    SEGMENT_LENGTH, STEP_SIZE

                )

                if num_of_subjects <= 2 and plot_preprocessing is not None and plot_fft is not None:
                    plot_preprocessing(ppg_signal, ppg_resampled, ppg_filtered, fs_orig=ORIGINAL_FS,
                                       fs_target=TARGET_FS, file_name=ppg_file, label='PPG', seconds=20)
                    plot_fft(ppg_resampled, TARGET_FS, label="PPG (Original)", filename=ppg_file, xlim=1.5)
                    plot_fft(ppg_filtered, TARGET_FS, label="PPG (Filtered)", filename=ppg_file, xlim=1.5)

                ppg_list.append(ppg_segments)
                resp_list.append(resp_segments)

    elif kind == 1:
        print("Processing BIDMC dataset...")
        data = scipy.io.loadmat(bidmc_mat_path)
        subjects = data['data'][0]
        for i in tqdm(range(len(subjects)), leave=True):
            try:
                ppg_signal = subjects[i]['ppg'][0][0][0].flatten()
                resp_signal = subjects[i]['ref']['resp_sig'][0][0][0][0][0][0][0][0].flatten()
                ppg_resampled,ppg_filtered,ppg_segments, resp_segments = process_signals(
                    ppg_signal, resp_signal,
                    125, TARGET_FS,
                    LOWCUT, HIGHCUT,
                    SEGMENT_LENGTH, STEP_SIZE
                )
                if ppg_segments is None or resp_segments is None:
                    print("it's none")
                ppg_list.append(ppg_segments)
                resp_list.append(resp_segments)
                num_of_subjects += 1
            except Exception as e:
                print(f"Error processing subject {i}: {e}")

    else:
        raise ValueError("Unknown dataset kind!")

    # Crop and stack as before
    min_len = min([sig.shape[0] for sig in ppg_list])
    ppg_list = [sig[:min_len] for sig in ppg_list]
    resp_list = [sig[:min_len] for sig in resp_list]
    data_ppg = np.stack(ppg_list, axis=0)
    data_resp = np.stack(resp_list, axis=0)

    print(f"Processed {num_of_subjects} subjects. data_ppg shape: {data_ppg.shape}, data_resp shape: {data_resp.shape}")
    return data_ppg, data_resp