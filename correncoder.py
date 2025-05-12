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


def plot_loss(train_losses):
    # Plotting the average epoch loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.title("Training Loss Over Epochs")
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


ppg_csv_path = "/Users/eli/Downloads/PPG Data/csv"
ppg_csv_files = [f for f in os.listdir(ppg_csv_path) if f.endswith('.csv') and not f.startswith('.DS_Store')]
input_name = 'PPG'
target_name = 'NASAL CANULA'

num_of_subjects = 0

original_fs = 256  # your original sampling rate
target_fs = 30  # your desired target rate

# define number of epochs and batch size
num_epochs = 50
batch_size = 16

# set a seed for evaluation (optional)
seed_val = 55
print("Seed")
print(seed_val)
torch.manual_seed(seed_val)
random.seed(seed_val)
np.random.seed(seed_val)

# set the learning rate for Adam optimisation
learning_rate = 0.0001

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

        # Segment the signals into non-overlapping windows (use segment length as 16*30)
        ppg_segments = segment_signal(ppg_signal, segment_length=60 * 30)
        resp_segments = segment_signal(resp_signal, segment_length=60 * 30)

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

kf = KFold(num_of_subjects)
kf.get_n_splits(data_ppg)
sub_num = 1

if np.any(np.isnan(data_ppg)):
    print(f"NaNs found in data_ppg")

if np.any(np.isnan(data_resp)):
    print(f"NaNs found in data_resp")


# Early stopping configuration
patience = 20

patience_counter = 0

# Store losses for plotting later

epsilon = 1e-8
for train_index, test_index in kf.split(data_ppg):
    trainX, testX = data_ppg[train_index, :], data_ppg[test_index, :]
    trainy, testy = data_resp[train_index, :], data_resp[test_index, :]

    for i in range(trainX.shape[0]):
        ppg_range = trainX[i].max() - trainX[i].min()
        if ppg_range < epsilon:
            print(f"Constant train_ppg signal at index {i}, replacing with zeros.")
            trainX[i] = np.zeros_like(trainX[i])
        else:
            trainX[i] = -1 + 2 * (trainX[i] - trainX[i].min()) / (
                        ppg_range + epsilon)  # This scales the signal to the range [-1, 1]

        resp_range = trainy[i].max() - trainy[i].min()
        if resp_range < epsilon:
            print(f"Constant train_resp signal at index {i}, replacing with zeros.")
            trainy[i] = np.zeros_like(trainy[i])
        else:
            trainy[i] = (trainy[i] - trainy[i].min()) / (
                        resp_range + epsilon)  # This rescales the signal to have a range between 0 and 1

    for i in range(testX.shape[0]):
        ppg_range = testX[i].max() - testX[i].min()
        if ppg_range < epsilon:
            print(f"Constant test_ppg signal at index {i}, replacing with zeros.")
            testX[i] = np.zeros_like(testX[i])
        else:
            testX[i] = -1 + 2 * (testX[i] - testX[i].min()) / (ppg_range + epsilon)

        resp_range = testy[i].max() - testy[i].min()
        if resp_range < epsilon:
            print(f"Constant test_resp signal at index {i}, replacing with zeros.")
            testy[i] = np.zeros_like(testy[i])
        else:
            testy[i] = (testy[i] - testy[i].min()) / (resp_range + epsilon)

    # print which subject is current test subject
    print("sub_num is: ", sub_num)

    # shuffle training data
    trainX, trainy = shuffle(trainX, trainy)

    # set a model path for saving the trained pytorch model weights
    model_path = "model_sub" + str(sub_num) + ".pth"

    # initialise new model
    model = Correncoder_model().to(device)  # Move model to the device
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ensure correct shape, can also transpose here instead of reshaping
    L_in = trainX.size // trainX.shape[0]
    # L_in = trainX.shape[-1]
    # Flatten subjects and segments into one batch dimension
    # Flatten subjects and segments into one batch dimension
    trainX = trainX.reshape(-1, trainX.shape[-1])  # shape: (total_segments, segment_length)
    trainX = trainX[:, np.newaxis, :]  # shape: (total_segments, 1, segment_length)
    trainy = trainy.reshape(-1, trainy.shape[-1])  # shape: (total_segments, segment_length)

    testX = testX.reshape(-1, testX.shape[-1])  # shape: (total_segments, segment_length)
    testX = testX[:, np.newaxis, :]  # shape: (total_segments, 1, segment_length)
    testy = testy.reshape(-1, testy.shape[-1])  # shape: (total_segments, segment_length)

    total_step = trainX.shape[0]

    # transformation of data into torch tensors
    trainXT = torch.from_numpy(trainX.astype('float32')).to(device)  # Convert to tensor and move to device
    trainyT = torch.from_numpy(trainy.astype('float32')).to(device)  # Convert to tensor and move to device
    testXT = torch.from_numpy(testX.astype('float32')).to(device)  # Convert to tensor and move to device
    testyT = torch.from_numpy(testy.astype('float32')).to(device)  # Convert to tensor and move to device
    # used for input to the breaths per minute calculator
    input_array = testyT.cpu().detach().numpy()

    loss_list = []
    acc_list = []
    acc_list_test_epoch = []
    test_error = []

    train_losses = []
    best_loss = float('inf')
    # begin training loop
    for epoch in range(num_epochs):
        epoch_loss = 0  # Track the loss for the entire epoch
        for i in tqdm(range(total_step // batch_size), desc=f"Epoch {epoch + 1}/{num_epochs}",
                      leave=True):  # split data into batches
            trainXT_seg = trainXT[i * batch_size:(i + 1) * batch_size, :, :]
            trainyT_seg = trainyT[i * batch_size:(i + 1) * batch_size]
            # Run the forward pass
            outputs = model(trainXT_seg)
            loss = criterion(outputs.squeeze(1), trainyT_seg)

            # Track the loss for the epoch
            epoch_loss += loss.item()

            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = epoch_loss / (total_step // batch_size)
        train_losses.append(avg_loss)
        tqdm.write(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

        # calculate test error at the end of each epoch
        model.eval()
        with torch.no_grad():
            test_output = model(testXT)
            loss_test = criterion(test_output, testyT[:, None])
            test_error.append((loss_test.item()))

        output_array = test_output.cpu().detach().numpy()

        mean_error_bpm = breaths_per_min_zc(output_array, input_array)

        print("Test sub")
        print(sub_num)
        print("Epoch")
        print(epoch)
        print("Test loss")
        print(loss_test)
        print("Peaks error abs")
        print(mean_error_bpm[0])
        print("Peaks error bias")
        print(mean_error_bpm[1])


        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")

                break

        model.train()

    # save the PyTorch model files
    torch.save(model.state_dict(), model_path)
    plot_loss(train_losses)
    random_indices = np.random.choice(len(output_array), size=1, replace=False)
    print(random_indices)
    for idx in random_indices:
        plt.figure(figsize=(20, 3))
        plt.plot(output_array[idx][0][:200], label='Predicted', color='blue')
        plt.plot(input_array[idx][:200], label='Ground Truth', color='red', linestyle='--')
        plt.title(f"Test Sample #{sub_num}")
        plt.xlabel("Time steps")
        plt.ylabel("Normalized Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.show()

    sub_num = sub_num + 1
