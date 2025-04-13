
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
import matplotlib.pyplot as plt

# ppg_csv_path = "/Users/eli/Downloads/PPG Data/csv"
ppg_csv_path = "/Users/elham/Downloads/csv/csv"
ppg_csv_files = [f for f in os.listdir(ppg_csv_path) if f.endswith('.csv') and not f.startswith('.DS_Store')]
input_name = 'PPG'
target_name = 'NASAL CANULA'

# Iterate over each file in the directory
for ppg_file in ppg_csv_files:
    # Load data
    ppg_df = pd.read_csv(os.path.join(ppg_csv_path, ppg_file), sep='\t', index_col='Time', skiprows=[1])

    print(f"Checking file: {ppg_file}")
    
    # Check if both input and target columns exist in the dataframe
    if input_name in ppg_df.columns and target_name in ppg_df.columns:
        # Extract the input and target columns
        input_signal = ppg_df[[input_name]].values.flatten()
        target_signal = ppg_df[[target_name]].values.flatten()

        # Get information about the PPG and AIRFLOW columns
        input_length = len(input_signal)
        target_length = len(target_signal)

        print(f"Length of input signal: {input_length}")
        print(f"Length of target signal: {target_length}")

        # Check for missing values (NaN) in both columns
        input_nan_count = ppg_df[input_name].isna().sum()
        target_nan_count = ppg_df[target_name].isna().sum()

        print(f"Missing values in input: {input_nan_count}")
        print(f"Missing values in target: {target_nan_count}")

        # Check if the lengths of both signals are the same
        if input_length != target_length:
            print(f"Warning: The lengths of input and target do not match. input length: {input_length}, target length: {target_length}")

        else:
            sampling_rate = 256
            time_limit = 5  # seconds

            # Calculate the number of samples corresponding to 30 seconds
            samples_to_plot = sampling_rate * time_limit

            # Plot the input and target signals on the same plot
            plt.figure(figsize=(12, 6))
            plt.plot(ppg_df.index[:samples_to_plot], input_signal[:samples_to_plot], label='Input Signal (PPG)', color='b')
            plt.plot(ppg_df.index[:samples_to_plot], target_signal[:samples_to_plot], label='Target Signal (NASAL CANULA)', color='r')
            plt.plot(ppg_df.index[:samples_to_plot], ppg_df['AIRFLOW'][:samples_to_plot], label='Airflow Signal', color='g')

            # Plot CHEST signal
            plt.plot(ppg_df.index[:samples_to_plot], ppg_df['CHEST'][:samples_to_plot], label='Chest Signal', color='y')

            # Plot ABDOMEN signal
            plt.plot(ppg_df.index[:samples_to_plot], ppg_df['ABDOMEN'][:samples_to_plot], label='Abdomen Signal', color='m')
            # Label the axes
            plt.xlabel('Time (s)')
            plt.ylabel('Signal Value')
            plt.title(f'Input and Output Signals for {ppg_file}')
            plt.legend()

            # Show the plot
            plt.show()
            exit()


    else:
        print("Both input and output columns are not present in this file.")

    print("-" * 40)  # Separator for readability