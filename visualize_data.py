
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

# ppg_df = pd.read_csv(os.path.join(ppg_csv_path, ppg_file), sep='\t', index_col='Time', skiprows=[1])
# print(ppg_df)
# exit()
# Iterate over each file in the directory
for ppg_file in ppg_csv_files:
    # Load data
    ppg_df = pd.read_csv(os.path.join(ppg_csv_path, ppg_file), sep='\t', index_col='Time', skiprows=[1])

    print(f"Checking file: {ppg_file}")
    
    # Check if both 'PPG' and 'AIRFLOW' columns exist in the dataframe
    if 'PPG' in ppg_df.columns and 'AIRFLOW' in ppg_df.columns:
        # Extract the PPG and AIRFLOW columns
        ppg_signal = ppg_df[['PPG']].values.flatten()
        airflow_signal = ppg_df[['AIRFLOW']].values.flatten()

        # Get information about the PPG and AIRFLOW columns
        ppg_length = len(ppg_signal)
        airflow_length = len(airflow_signal)

        print(f"Length of PPG signal: {ppg_length}")
        print(f"Length of AIRFLOW signal: {airflow_length}")

        # Check for missing values (NaN) in both columns
        ppg_nan_count = ppg_df['PPG'].isna().sum()
        airflow_nan_count = ppg_df['AIRFLOW'].isna().sum()

        print(f"Missing values in PPG: {ppg_nan_count}")
        print(f"Missing values in AIRFLOW: {airflow_nan_count}")

        # Check if the lengths of both signals are the same
        if ppg_length == airflow_length:
            print("The lengths of PPG and AIRFLOW are the same.")
        else:
            print(f"Warning: The lengths of PPG and AIRFLOW do not match. PPG length: {ppg_length}, AIRFLOW length: {airflow_length}")

    else:
        print("Both PPG and AIRFLOW columns are not present in this file.")

    print("-" * 40)  # Separator for readability