import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import pickle
from concurrent.futures import ProcessPoolExecutor
from sklearn.utils import shuffle
import random

fixed_timesteps = 1001

def get_length(file_path):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    return mfccs.shape[1]

def determine_fixed_length(directory):
    file_paths = []

    for subdir in ['1_4', '2_4', '3_4', '4_4']:
        for file in tqdm(os.listdir(os.path.join(directory, subdir))):
            file_path = os.path.join(directory, subdir, file)
            file_paths.append(file_path)

    # Utilize multiprocessing for faster computation
    with ProcessPoolExecutor() as executor:
        lengths = list(executor.map(get_length, file_paths))

    return min(lengths)

def parallel_data_loader(directories):
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(parallel_load_and_preprocess, directories), total=len(directories)))
    return results

def load_and_preprocess_data_subdir(args):
    directory, subdir = args
    data = []
    labels = []
    
    for file in os.listdir(os.path.join(directory, subdir)):
        file_path = os.path.join(directory, subdir, file)
        processed_data = preprocess_audio(file_path)
        data.append(processed_data)
        label = ['1_4', '2_4', '3_4', '4_4'].index(subdir)
        labels.append(label)
    
    return data, labels

def parallel_load_and_preprocess(directory):
    # Create a pool of processes
    pool = Pool(cpu_count())

    # Create a list of tasks
    tasks = [(directory, time_sig) for time_sig in ['1_4', '2_4', '3_4', '4_4']]

    # Use imap_unordered to distribute the work among the processes
    results = list(tqdm(pool.imap_unordered(load_and_preprocess_data_subdir, tasks), total=len(tasks), mininterval=0.01))

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    # Combine results
    combined_data = []
    combined_labels = []
    
    for data, labels in results:
        combined_data.extend(data)
        combined_labels.extend(labels)
    
    return combined_data, combined_labels


def adjust_fixed_length(features, timesteps):
    # If the array is 1-dimensional
    if len(features.shape) == 1:
        if features.shape[0] > timesteps:
            return features[:timesteps]
        elif features.shape[0] < timesteps:
            padding = np.zeros(timesteps - features.shape[0])
            return np.hstack((features, padding))
        return features
    # If the array is 2-dimensional
    else:
        # If the time axis of the 2D array is greater than timesteps, crop it.
        if features.shape[1] > timesteps:
            return features[:, :timesteps]
        # If the time axis of the 2D array is less than timesteps, pad it.
        elif features.shape[1] < timesteps:
            padding = np.zeros((features.shape[0], timesteps - features.shape[1]))
            return np.hstack((features, padding))
        return features

# Convert real-valued features to Poisson spike trains
def poisson_spike_encoding(data, duration=10, dt=1*ms):
    # Assuming data is normalized between 0 and 1
    rates = data * (1.0/dt)
    spikes = (np.random.rand(*data.shape) < rates*dt).astype(float)
    return spikes

def temporal_binning(data, bin_size):
    """
    Bins the data into chunks of bin_size and returns the average of each chunk.
    """
    # Split the data into chunks of bin_size
    binned_data = [np.mean(data[i:i+bin_size]) for i in range(0, len(data), bin_size)]
    return np.array(binned_data)

def rate_based_encoding(data, min_freq, max_freq):
    """
    Convert onset strengths to spike frequencies.
    data: The input data (should be normalized to [0, 1])
    min_freq: The minimum spike frequency (corresponds to data value of 0)
    max_freq: The maximum spike frequency (corresponds to data value of 1)
    Returns: Spike frequencies corresponding to input data
    """
    return min_freq + data * (max_freq - min_freq)

def extract_bpm_and_instrument(file_path):
    match = re.search(r"instrument_(\d+)_bpm_(\d+)", file_path)
    if match:
        instrument = match.group(1)
        bpm = match.group(2)
        return instrument, bpm
    return None, None

def moving_average(data, window_size):
    """Compute moving average"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


# Process the audio file into desired features
# Process the audio file into desired features
def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)  # setting sr ensures all files are resampled to this rate
    time_signature = file_path.split('/')[-2].replace('_', '/')
    instrument, bpm = extract_bpm_and_instrument(file_path)

    # Extracting onset strength
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
    
    # Extracting tempogram
    tempogram = librosa.feature.tempogram(onset_envelope=onset_strength, sr=sr)
    
    # Extracting tempogram
    tempogram_cropped = librosa.feature.tempogram(onset_envelope=onset_strength[20:], sr=sr)
    
    # Adjust the time axis of each feature to fixed_timesteps
    onset_strength_fixed = adjust_fixed_length(onset_strength, fixed_timesteps)
    tempogram_fixed = adjust_fixed_length(tempogram, fixed_timesteps)

    # Stacking features horizontally
    combined_features = np.vstack(poisson_spike_encoding(onset_strength))
    
    # Normalize to range [0, 1]
    encoded_features = (combined_features - np.min(combined_features)) / (np.max(combined_features) - np.min(combined_features))
    
        # Plotting
    plt.figure(figsize=(12, 14))
    plt.title('audio  with {time_signature} time signature, {bpm} bpm, and instrument {instrument}')

    rows = 6
    # 1. Raw audio
    plt.subplot(rows, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Raw Audio')

    # 2. Onset strength
    plt.subplot(rows, 1, 2)
    plt.plot(onset_strength_fixed)
    plt.title('Onset Strength fixed size')
    
    # 2. Onset strength
    plt.subplot(rows, 1, 3)
    onset_strength_normalized = (onset_strength[20:] - np.min(onset_strength[20:])) / (np.max(onset_strength[20:]) - np.min(onset_strength[20:]))
    plt.plot(onset_strength_normalized)
    plt.title('Onset Strength normalized and cropped')
    
    # Add a plot for averaged onset strength
    plt.subplot(rows, 1, 4)
    averaged_onset = moving_average(onset_strength_normalized, window_size=5)  # using a window size of 10, adjust as needed
    plt.plot(averaged_onset)
    plt.title('Averaged Onset Strength')
    
    # 3. Tempogram
    plt.subplot(rows, 1, 5)
    librosa.display.specshow(tempogram_fixed, sr=sr, x_axis='time', y_axis='tempo')
    plt.title('Tempogram fixed')
    
        # 3. Tempogram
    plt.subplot(rows, 1, 6)
    librosa.display.specshow(tempogram_cropped, sr=sr, x_axis='time', y_axis='tempo')
    plt.title('Tempogram cropped')
    
    
    
    plt.tight_layout()
    plt.savefig(f'output_processing_noise_avg/{time_signature.replace("/", "_")}_BPM{bpm}_noise.png')
    
    return encoded_features[20:]


def count_files(directory):
    return sum([len(files) for _, _, files in os.walk(directory)])

# Current directory
directory = '.'

# Loop through all files in the current directory
for filename in os.listdir(directory):
    # Check if the filename ends with '.png' and contains 'spike_train'
    if filename.endswith('.png') and 'spike_train' in filename:
        # Construct the full file path
        filepath = os.path.join(directory, filename)
        
        # Remove the file
        os.remove(filepath)
        print(f"Deleted: {filename}", end='\r')
        

# checking shapes
print("Checking shapes...")
fixed_timesteps = determine_fixed_length('training_data')
print(fixed_timesteps)
fixed_timesteps2 = determine_fixed_length('validation_data')
print(fixed_timesteps2)
fixed_timesteps = max(fixed_timesteps, fixed_timesteps2)


# 1. Load and preprocess data
print("Loading and preprocessing training data...")
directories = ['training_data', 'validation_data']
training_data_results, validation_data_results = parallel_data_loader(directories)

training_data, training_labels = training_data_results
validation_data, validation_labels = validation_data_results
print("\nDone with preprocessing!")



