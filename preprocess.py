import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, decimate
import pickle



# ppg_csv_path = "/Users/eli/Downloads/PPG Data/csv"
ppg_csv_path = "/Users/elham/Downloads/csv/csv"
ppg_csv_files = [f for f in os.listdir(ppg_csv_path) if f.endswith('.csv') and not f.startswith('.DS_Store')]

# gold_standard_path = "/Users/eli/Downloads/Gold-Standard"
gold_standard_path = "/Users/elham/Downloads/Gold-Standard/Gold-Standard"
gold_files = [f for f in os.listdir(gold_standard_path) if f.endswith(".xlsx") and not f.startswith(".")]



def fill_missing_gold():
    for f in gold_files:
        file_path = os.path.join(gold_standard_path, f)
        print(f)
        
        # Read the data from the Excel file
        gold_df = pd.read_excel(f"{gold_standard_path}/{f}")
        print(gold_df['RR count, actual'])
        
        # Replace non-numeric values (like '-') with NaN
        gold_df['RR count, actual'] = gold_df['RR count, actual'].replace("-", pd.NA)

        # Convert to numeric (useful if it was read as strings)
        gold_df['RR count, actual'] = pd.to_numeric(gold_df['RR count, actual'], errors="coerce")
        
        # Step 1: Find the last valid (non-NaN) value in the column
        last_valid_index = gold_df['RR count, actual'].last_valid_index()
        
        # Step 2: Calculate the mean of the valid values before the last valid row
        valid_values_before_last = gold_df['RR count, actual'][:last_valid_index].dropna()
        mean_value = valid_values_before_last.mean()

        # Step 3: Fill missing values before the last valid row with the mean value
        gold_df['RR count, actual'].fillna(value=mean_value, inplace=True)

        # Step 4: Remove rows after the last valid value
        gold_df = gold_df.loc[:last_valid_index]

        # Step 5: Save the updated data (overwrite original file)
        gold_df.to_excel(file_path, index=False)

        print(gold_df['RR count, actual'])
fill_missing_gold()

paired_files = {}

for ppg_file in ppg_csv_files:
    subject_id = ppg_file[:2]  # Extract first two characters
    matched_gold = [gf for gf in gold_files if gf.startswith(subject_id)]

    if matched_gold:
        paired_files[ppg_file] = matched_gold[0]  # Assuming one match per subject


# Define the path to save the pickle file
# pickle_file_path = "/Users/eli/Downloads/paired_files.pkl"  # Modify the path as needed
# pickle_file_path = "/Users/elham/Downloads/paired_files.pkl"  # Modify the path as needed

# with open(pickle_file_path, 'wb') as pickle_file:
#     pickle.dump(paired_files, pickle_file)

# print(f"Dictionary saved to {pickle_file_path}")


# with open(pickle_file_path, 'rb') as pickle_file:
#     paired_files = pickle.load(pickle_file)

# print(f"Dictionary loaded: {paired_files}")


# Kalman Filter Implementation (for signal denoising)
class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, initial_value, initial_estimate_error):
        self.process_variance = process_variance  # Variance of the process noise
        self.measurement_variance = measurement_variance  # Variance of the measurement noise
        self.estimated_value = initial_value  # Initial estimate of the state (signal)
        self.estimate_error = initial_estimate_error  # Initial estimate error

    def update(self, measurement):
        # Prediction step
        predicted_value = self.estimated_value
        predicted_error = self.estimate_error + self.process_variance

        # Update step (correct based on the actual measurement)
        kalman_gain = predicted_error / (predicted_error + self.measurement_variance)
        self.estimated_value = predicted_value + kalman_gain * (measurement - predicted_value)
        self.estimate_error = (1 - kalman_gain) * predicted_error

        return self.estimated_value


def bandpass_filter(signal, lowcut=0.1, highcut=0.6, fs=256, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def downsample(signal, target_fs=30, fs=256):
    downsample_factor = int(fs / target_fs)
    return signal[::downsample_factor]



# Main function for processing the paired PPG and Gold standard files
def process_ppg_data(window_size_seconds=60, srate=256, target_fs=30):
    # Convert window size in seconds to number of samples based on the original sampling rate
    window_size_samples = window_size_seconds * target_fs  # Window size in samples for original srate

    # Initialize scalers
    scaler_ppg = MinMaxScaler(feature_range=(-1, 1))  # Scaling to range [-1, 1]
    scaler_resp = MinMaxScaler(feature_range=(0, 1))   # Scaling to range [0, 1] for respiratory rate
    
    X, Y = [], []
    
    # Define the path to load the pickle file for paired files
    # pickle_file_path = "/Users/eli/Downloads/paired_files.pkl"  # Modify the path as needed
    pickle_file_path = "/Users/elham/Downloads/paired_files.pkl"  # Modify the path as needed

    with open(pickle_file_path, 'rb') as pickle_file:
        paired_files = pickle.load(pickle_file)

    print(f"Dictionary loaded: {paired_files}")

    for ppg_file, gold_file in paired_files.items():
        # Load data
        ppg_df = pd.read_csv(os.path.join(ppg_csv_path, ppg_file), sep='\t', index_col='Time', skiprows=[1])
        gold_df = pd.read_excel(os.path.join(gold_standard_path, gold_file))

        ppg_signal = ppg_df[['PPG']].values.flatten()
        resp_signal = gold_df[['RR count, actual']].values.flatten()
        print(len(resp_signal))
        print(len(ppg_signal))

        # Step 1: Bandpass filtering to isolate respiratory signal (0.1-0.6 Hz)
        ppg_filtered = bandpass_filter(ppg_signal, lowcut=0.1, highcut=0.6, fs=srate)

        # Step 2: Kalman Filtering (motion artifact removal)
        kalman_filter = KalmanFilter(process_variance=1e-5, measurement_variance=1e-1, initial_value=0, initial_estimate_error=1)
        ppg_denoised = np.array([kalman_filter.update(x) for x in ppg_filtered])

        # Step 3: Scaling the PPG signal to [-1, 1]
        ppg_scaled = scaler_ppg.fit_transform(ppg_denoised.reshape(-1, 1)).flatten()

        # Step 4: Interpolation of the respiratory signal to match the PPG length
        resp_scaled = scaler_resp.fit_transform(resp_signal.reshape(-1, 1)).flatten()
        

        # Step 5: Resample PPG to target_fs (downsample from srate to target_fs)
        ppg_resampled = downsample(ppg_scaled, target_fs=target_fs, fs=srate)
        diff = len(ppg_resampled)/window_size_samples - len(resp_scaled)
        diff*window_size_samples
        # Ensure segmentation is possible
        if len(ppg_resampled) >= window_size_samples:  # Ensure we have enough data for segmentation
            for i in range(0, len(ppg_resampled) - window_size_samples + 1, window_size_samples):  # Non-overlapping windows
                X.append(ppg_resampled[i:i + window_size_samples])  # Segments
                minute_index = i // window_size_samples  # Calculate the minute index
                Y.append(resp_scaled[minute_index])  # Append the corresponding respiratory rate
        else:
            print(f"Skipping interval due to short signal: {len(ppg_resampled)} samples.")

        

    # Ensure all sequences are of the same shape
    X = np.array([x for x in X if len(x) == window_size_samples]).reshape(-1, window_size_samples, 1)  # (num_samples, window_size, 1)
    Y = np.array([y for y in Y if len(y) == window_size_samples]).reshape(-1, window_size_samples, 1)

    # Train-Test Split
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
    # Second split: Validation (10%) and Test (10%) from Temp (20%)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

# Example of running the function
X_train, X_val, X_test, Y_train, Y_val, Y_test = process_ppg_data(window_size_seconds=60, srate=256, target_fs=30)