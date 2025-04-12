import wfdb
import os


file_path = '/Users/elham/Downloads/Telegram Desktop/Respilife PZ-BR01-13 011719PZ2 (01-17-2019).REC'
if os.path.exists(file_path):
    print(f"File exists: {file_path}")# Read the .rec file and header
else:
    print(f"File not found: {file_path}")
# record = wfdb.rdrecord(file_path)






import pandas as pd

# Try reading the .rec file as a CSV file
df = pd.read_csv(file_path)

# Check the first few rows
print(df.head())


# # Print out the information about the signal
# print(record.__dict__)

# # To access the signal data (e.g., PPG or ECG signal)
# signal = record.p_signal
# print(signal)
