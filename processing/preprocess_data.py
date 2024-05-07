import pandas as pd
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt
from tqdm import tqdm

def load_raw_data(df, sampling_rate, path):
    data = []
    if sampling_rate == 100:
        for f in tqdm(df.filename_lr, desc="Loading data"):
            data.append(wfdb.rdsamp(path+f))
    else:
        for f in tqdm(df.filename_hr, desc="Loading data"):
            data.append(wfdb.rdsamp(path+f))
    data = np.array([signal for signal, meta in data])
    return data

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)

# Load and preprocess the data
path = 'data/'
sampling_rate = 100
cutoff_frequency = 40

# Load and convert annotation data (for file names only)
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# Apply the low-pass filter to the ECG data
X_filtered = np.zeros_like(X)
for i in tqdm(range(X.shape[0]), desc="Filtering data"):
    for j in range(X.shape[2]):
        X_filtered[i, :, j] = apply_lowpass_filter(X[i, :, j], cutoff_frequency, sampling_rate)

# Save the filtered data as a new file
np.save('data/ptbxl_filtered_data.npy', X_filtered)