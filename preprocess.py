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

ppg_csv_path = "/Users/eli/Downloads/PPG Data/csv"
ppg_csv_files = [f for f in os.listdir(ppg_csv_path) if f.endswith('.csv') and not f.startswith('.DS_Store')]

gold_standard_path = "/Users/eli/Downloads/Gold-Standard"
gold_files = [f for f in os.listdir(gold_standard_path) if f.endswith(".xlsx") and not f.startswith(".")]


# paired_files = {}
#
# for ppg_file in ppg_csv_files:
#     subject_id = ppg_file[:2]  # Extract first two characters
#     matched_gold = [gf for gf in gold_files if gf.startswith(subject_id)]
#
#     if matched_gold:
#         paired_files[ppg_file] = matched_gold[0]  # Assuming one match per subject


# Define the path to save the pickle file
pickle_file_path = "/Users/eli/Downloads/paired_files.pkl"  # Modify the path as needed

# with open(pickle_file_path, 'wb') as pickle_file:
#     pickle.dump(paired_files, pickle_file)
#
# print(f"Dictionary saved to {pickle_file_path}")


with open(pickle_file_path, 'rb') as pickle_file:
    paired_files = pickle.load(pickle_file)

print(f"Dictionary loaded: {paired_files}")


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