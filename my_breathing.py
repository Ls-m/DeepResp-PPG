import csv
import torch
import pandas as pd
from scipy.signal import butter, filtfilt
import os
from scipy.signal import resample
import numpy as np


ppg_csv_path = "/Users/elham/Downloads/csv/csv"
ppg_csv_files = [f for f in os.listdir(ppg_csv_path) if f.endswith('.csv') and not f.startswith('.DS_Store')]
input_name = 'PPG'
target_name = 'NASAL CANULA'


fs = 30

cutoff = 1  # Desired cutoff frequency of the filter (Hz)
order = 8  # Order of the filter

# Design Butterworth low-pass filter
nyquist = 0.5 * fs  # Nyquist Frequency
normal_cutoff = cutoff / nyquist  # Normalize the frequency
b, a = butter(order, normal_cutoff, btype='low', analog=False)

ppg_whole = []
resp_whole = []
seg_length = 5

num_of_subjects = 0
# data = scipy.io.loadmat('bidmc_data.mat')
windowed_pleth_list = []
windowed_resp_list = []


# Iterate over each file in the directory
for ppg_file in ppg_csv_files:
    data = pd.read_csv(os.path.join(ppg_csv_path, ppg_file), sep='\t', index_col='Time', skiprows=[1])
    if input_name in data.columns and target_name in data.columns:
        num_of_subjects += 1
        ppg_sub = []
        resp_sub = []
        ppg = data[input_name].to_numpy().reshape(-1, 1)
        print("type ppg is: ",type(ppg))
        resp = data[target_name].to_numpy().reshape(-1, 1)
        print("type resp is: ",type(resp))
        print("shape 1 resp (raw) is: ",resp.shape[0])
        target_freq = 30
        coef = target_freq/256
        print("coef is: ",coef)
        resp = resample(resp, int(resp.shape[0]*coef))
        ppg = resample(ppg, int(ppg.shape[0]*coef))
        print("shape 2 resp (resampled) is: ",resp.shape[0])
        resp = filtfilt(b, a, resp[:,0])
        print("shape 3 resp (low pass filtered) is : ",resp.shape[0])
        for seg in range(int(np.floor(resp.shape[0]/(seg_length*30)))):
            resp_sub.append(resp[None,seg_length*30*seg:seg_length*30*(seg+1),None])
            ppg_sub.append(ppg[None,seg_length*30*seg:seg_length*30*(seg+1)])

        windowed_pleth =  np.concatenate(ppg_sub, axis =0).transpose(2,0,1)
        windowed_resp =  np.concatenate(resp_sub, axis =0).transpose(2,0,1)

        windowed_pleth_list.append(windowed_pleth)
        windowed_resp_list.append(windowed_resp)


resp  = np.concatenate(windowed_resp_list, axis = 0)
ppg  = np.concatenate(windowed_pleth_list, axis = 0)


class SimpleCSVLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        # If the file doesn't exist, create one with headers.
        if not os.path.exists(filepath):
            with open(filepath, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['subject', 'metric1', 'value1', 'metric2', 'value2', 'metric3', 'value3'])

    def log(self, subject, metric1, value1, metric2, value2, metric3, value3):
        with open(self.filepath, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([subject, metric1, value1, metric2, value2, metric3, value3])

# device = torch.device("cuda:2")
device = torch.device("cpu")
overall_breathing = 0
overall_mae = 0

print("number of subjects is: ",num_of_subjects)