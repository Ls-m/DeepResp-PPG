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
        # breaths_per_min_output = (zero_crossings_output / 2)*6.25
    peak_count_output = np.array(peak_count_output)
    peak_count_cap = np.array(peak_count_cap)
    #6.5 is used ot scale up to 1 minute, as each segment here is 60/6.5 seconds long.
    mean_error = ((np.mean(peak_count_output - peak_count_cap)) / 2) * 6.5
    mean_abs_error = ((np.mean(np.abs(peak_count_output - peak_count_cap))) / 2) * 6.5
    return mean_abs_error, mean_error


# ppg_csv_path = "/Users/eli/Downloads/PPG Data/csv"
ppg_csv_path = "/Users/elham/Downloads/csv/csv"

ppg_csv_files = [f for f in os.listdir(ppg_csv_path) if f.endswith('.csv') and not f.startswith('.DS_Store')]
input_name = 'PPG'
target_name = 'NASAL CANULA'

num_of_subjects = 0



# define number of epochs and batch size
num_epochs = 80
batch_size = 8

#define number of kernels per layer
n_in, n_out = 1, 8
n_out2 = 8
n_out3 = 8
n_outputs = 1

# define kernel lengths, padding, dilation, stride, and dropout
kernel_size = 150
kernel_size2 = 75
kernel_size3 = 50
padding = 20
dilation = 1
stride = 1
dropout_val = 0.5
padding2 = 20
padding3 = 10
dilation2 = 1
dilation3 = 1
stride2 = 1
stride3 = 1

# set a seed for evaluation (optional)
seed_val = 55
print("Seed")
print(seed_val)
torch.manual_seed(seed_val)
random.seed(seed_val)
np.random.seed(seed_val)

# set the learning rate for Adam optimisation
learning_rate = 0.001





ppg_list = []
resp_list = []

for ppg_file in tqdm(ppg_csv_files[:10], leave=True) :
    data = pd.read_csv(os.path.join(ppg_csv_path, ppg_file), sep='\t', index_col='Time', skiprows=[1])
    if input_name in data.columns and target_name in data.columns:
        num_of_subjects += 1
        ppg_signal = data[input_name].to_numpy()  # extract the PPG signal
        resp_signal = data[target_name].to_numpy()  # extract the target signal (NASAL CANULA, AIRFLOW, etc.)


        # append each subject's signal to the list
        ppg_list.append(ppg_signal)
        resp_list.append(resp_signal)


# After looping over all subjects, stack them together
data_ppg = np.stack(ppg_list, axis=0)  # shape: (num_subjects, signal_length)
data_resp = np.stack(resp_list, axis=0)  # shape: (num_subjects, signal_length)

print(f"data_ppg shape: {data_ppg.shape}")
print(f"data_co2 shape: {data_resp.shape}")


kf = KFold(42)
kf.get_n_splits(data_ppg)
sub_num = 1


for train_index, test_index in kf.split(data_ppg):
    trainX, testX = data_ppg[train_index, :], data_ppg[test_index, :]
    trainy, testy = data_resp[train_index, :], data_resp[test_index, :]

    # print which subject is current test subject
    print(sub_num)

    # shuffle training data
    trainX, trainy = shuffle(trainX, trainy)

    # set a model path for saving the trained pytorch model weights
    model_path = "model_sub" + str(sub_num) + ".pth"

    # initialise new model
    model = Correncoder_model()
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # ensure correct shape, can also transpose here instead of reshaping
    L_in = trainX.shape[-1]
    trainX = trainX.reshape((trainX.shape[0], 1, L_in))
    testX = testX.reshape((testX.shape[0], 1, L_in))

    total_step = trainX.shape[0]

    # transformation of data into torch tensors
    trainXT = torch.from_numpy(trainX.astype('float32'))
    # trainXT = trainXT.transpose(1,2).float() #input is (N, Cin, Lin) = Ntimesteps, Nfeatures, 128
    trainyT = torch.from_numpy(trainy.astype('float32'))
    testXT = torch.from_numpy(testX.astype('float32'))

    testyT = torch.from_numpy(testy.astype('float32'))
    # used for input to the breaths per minute calculator
    input_array = testyT.cpu().detach().numpy()

    loss_list = []
    acc_list = []
    acc_list_test_epoch = []
    test_error = []

    #begin training loop
    for epoch in range(num_epochs):
        for i in range(total_step // batch_size):  # split data into batches
            trainXT_seg = trainXT[i * batch_size:(i + 1) * batch_size, :, :]
            trainyT_seg = trainyT[i * batch_size:(i + 1) * batch_size, None]
            # Run the forward pass
            outputs = model(trainXT_seg)

            loss = criterion(outputs, trainyT_seg)

            #loss = d_loss_output[0]

            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
        # calculate test error at the end of each epoch
        test_output = model(testXT)
        loss_test = criterion(test_output, testyT[:, None])
        test_error.append((loss_test.item()))

        output_array = test_output.cpu().detach().numpy()

        mean_error_bpm = breaths_per_min_zc(output_array, input_array)

        print("Test sub")
        print(sub_num)
        print("Epoch")
        print(epoch)
        print("Training loss")
        print(loss)
        print("Test loss")
        print(loss_test)
        print("Peaks error abs")
        print(mean_error_bpm[0])
        print("Peaks error bias")
        print(mean_error_bpm[1])

    # save the PyTorch model files
    torch.save(model.state_dict(), model_path)
    sub_num = sub_num + 1

