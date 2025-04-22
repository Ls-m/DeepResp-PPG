import csv
import torch
import pandas as pd
from scipy.signal import butter, filtfilt
import os
from scipy.signal import resample
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import diffusion_pipeline
from torch.optim import Adam
from tqdm import tqdm
from scipy.fftpack import fft

def calculate_respiratory_rate(signal, sampling_rate):
    """
    Calculate the respiratory rate from a signal using FFT to find the most dominant frequency.

    :param signal: The input signal, a 1D numpy array.
    :param sampling_rate: The sampling rate of the signal in Hz.
    :return: The respiratory rate in breaths per minute (BPM).
    """

    # Length of the signal
    n = len(signal)

    # Perform FFT on the signal
    fft_values = fft(signal)

    # Frequency bins
    freq = np.fft.fftfreq(n, d=1 / sampling_rate)

    # Get the magnitude of the FFT
    fft_magnitude = np.abs(fft_values)

    # Consider only the positive half of the frequencies
    positive_freq_idx = np.where(freq > 0)
    freq = freq[positive_freq_idx]
    fft_magnitude = fft_magnitude[positive_freq_idx]

    # Find the peak frequency
    peak_frequency = freq[np.argmax(fft_magnitude)]

    # Convert frequency to respiratory rate (breaths per minute)
    respiratory_rate_bpm = peak_frequency * 60

    return respiratory_rate_bpm


ppg_csv_path = "./csv"
# ppg_csv_path = "/Users/elham/Downloads/csv/csv"
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
min_len = 1e9

# Iterate over each file in the directory
# Limit the number of files to process (e.g., first 5 files)
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
        min_len = min(len(windowed_resp[0]), min_len)


resp = [resp.squeeze(axis=0)[:min_len, :] for resp in windowed_resp_list]
# resp  = np.squeeze(windowed_resp_list, axis = 1)

# resp  = np.concatenate(windowed_resp_list, axis = 0)
ppg = [ppg.squeeze(axis=0)[:min_len, :] for ppg in windowed_pleth_list]

# ppg  = np.concatenate(windowed_pleth_list, axis = 0)


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
device = torch.device("cuda")

overall_breathing = 0
overall_mae = 0

print("number of subjects is: ",num_of_subjects)


class Diff_dataset(Dataset):
    def __init__(self, ppg, co2):
        self.ppg = ppg
        self.co2 = co2

    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, index):
        return torch.tensor(self.ppg[index, None, :], dtype=torch.float32), torch.tensor(self.co2[index, None, :],
                                                                                         dtype=torch.float32)


# Loop over each trial and use it as the test set
for subject_id in range(num_of_subjects):

    test_ppg = ppg[subject_id]        # Extract the i-th trial as the test set, shape (95, 300)
    train_ppg = np.delete(ppg, subject_id, axis=0)  # Remove the i-th trial, shape (41, 95, 300)
    test_resp = resp[subject_id]        # Extract the i-th trial as the test set, shape (95, 300)
    train_resp = np.delete(resp, subject_id, axis=0)  # Remove the i-th trial, shape (41, 95, 300)

    train_resp = train_resp.reshape(-1,train_resp.shape[-1])
    train_ppg = train_ppg.reshape(-1,train_ppg.shape[-1])

    
    # Apply the filter to the signal

    for i in range(train_ppg.shape[0]):
        train_ppg[i] = -1 + 2*(train_ppg[i] - train_ppg[i].min())/(train_ppg[i].max() - train_ppg[i].min())
        train_resp[i] = (train_resp[i] - train_resp[i].min())/(train_resp[i].max() - train_resp[i].min())


    for i in range(test_ppg.shape[0]):
        test_ppg[i] = -1 + 2*(test_ppg[i] - test_ppg[i].min())/(test_ppg[i].max() - test_ppg[i].min())
        test_resp[i] = (test_resp[i] - test_resp[i].min())/(test_resp[i].max() - test_resp[i].min())






    train_dataset = Diff_dataset(train_ppg, train_resp)
    val_dataset = Diff_dataset(test_ppg, test_resp)

    train_loader = DataLoader(train_dataset, batch_size=128,shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size=64,shuffle = False)




    model =     model = diffusion_pipeline(384, 1024, 6, 128, device).to(device)
    #model = torch.nn.DataParallel(model).to(device)

    

    import time
    start = time.time()
    optimizer = Adam(model.parameters(), lr=1e-4)
    log_path = os.path.join('model_5s_double_final_corrected.csv')
    logger = SimpleCSVLogger(log_path)
    num_epochs = 400
    best_val_loss = 10000000000
    p1 = int(0.7 * num_epochs)
    p2 = int(0.99 * num_epochs)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1)
    


    for epoch_no in range(num_epochs):
            avg_loss = 0
            model.train()
            with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_no, train_batch in enumerate(it, start=1):
                    optimizer.zero_grad()
                    loss = model(train_batch[0].to(device), co2 = train_batch[1].to(device),  flag = 0)
                    loss = loss.mean(dim = 0)
                    loss.backward()
                    avg_loss += loss.item()/train_batch[0].shape[0]
                    optimizer.step()
                    it.set_postfix(
                        ordered_dict={
                            "Train: avg_epoch_loss": avg_loss / batch_no,
                            "epoch": epoch_no,
                        },
                        refresh=False,
                    )

                lr_scheduler.step()



                
    output_path = os.path.join(f'model_bi{subject_id}_final.pth')
    torch.save(model.state_dict(), output_path)
    end = time.time()
    print('overall time: ', (start - end)/60)    
    mae = 0
    num_windows = 0
    
    model.eval()

    results = []

    with tqdm(val_loader, mininterval=5.0, maxinterval=50.0) as it:
        for batch_no, val_batch in enumerate(it, start=1):
            y = model(val_batch[0].to(device), n_samples=100, flag=1)
            r = y[:,0:100,0,:].mean(dim=1).detach().cpu().numpy()
            results.append(y)
            num_windows = num_windows + val_batch[0].shape[0]

            for i in range(val_batch[0].shape[0]):
                truth = val_batch[1][i,0,:].detach().cpu().numpy()
                mae_current = np.abs(r[i] - truth).mean()
                mae = mae + mae_current

    print('mae: ', mae/num_windows)
    for i in range(len(results)):
        results[i] = results[i][:,0:100,0,:].mean(dim=1).detach().cpu().numpy()
    segment_results = np.concatenate(results, axis = 0)

    whole_trial_resp = []
    segment_resp = test_resp
    for i in range(0, segment_resp.shape[0]):
        whole_trial_resp.append(segment_resp[i])
    whole_trial_resp = np.concatenate(whole_trial_resp, axis=0)

    segment_results2 = segment_results

    whole_trial_results = []
    for i in range(0, segment_results.shape[0]):
        whole_trial_results.append(segment_results[i])
    whole_trial_results = np.concatenate(whole_trial_results, axis=0)
    whole_trial_results = filtfilt(b, a, whole_trial_results)
    whole_trial_ppg = []
    test_ppg = test_ppg
    segment_ppg = test_ppg
    for i in range(0, segment_ppg.shape[0]):
        segment_ppg[i] = (segment_ppg[i] - segment_ppg[i].min()) / (segment_ppg[i].max() - segment_ppg[i].min())
        whole_trial_ppg.append(segment_ppg[i])
    whole_trial_ppg = np.concatenate(whole_trial_ppg, axis=0)

    E = 0
    dc_truth = []
    dc_results = []
    overlap = 600
    for i in range(int(np.floor((whole_trial_results.shape[0] - 1800) / overlap)) + 1):
        c_current = np.zeros((1800))
        r_current = np.zeros((1800))
        r_current = whole_trial_results[overlap * i:overlap * i + 1800].copy()
        c_current = whole_trial_resp[overlap * i:overlap * i + 1800].copy()


        RR_truth = calculate_respiratory_rate(c_current, 30)
        RR_predicted = calculate_respiratory_rate(r_current, 30)
        E = E + np.abs(RR_truth - RR_predicted)

        c_current[c_current >= 0.5] = 1
        c_current[c_current < 0.5] = 0
        r_current[r_current >= 0.5] = 1
        r_current[r_current < 0.5] = 0

        dc_truth.append(c_current.sum())
        dc_results.append(r_current.sum())

    print('Mean Breathing Rate Difference: ', E / int(np.floor((whole_trial_results.shape[0] - 1800) / overlap)))
    print('Whole Trial MAE: ', np.abs(whole_trial_results - whole_trial_resp).mean())
    cor = np.corrcoef(dc_truth, dc_results)
    print('duty cycle correlation:', cor)
    breathing_diff = E / int(np.floor((whole_trial_results.shape[0] - 1800) / overlap))
    mae_whole = np.abs(whole_trial_results - whole_trial_resp).mean()
    logger.log(subject_id, 'breathing', breathing_diff, 'mae_whole', mae_whole, 'corr', cor[0, 1])
    overall_breathing = overall_breathing + breathing_diff
    overall_mae = overall_mae + mae_whole
print(overall_breathing / num_of_subjects)
print(overall_mae / num_of_subjects)


    
