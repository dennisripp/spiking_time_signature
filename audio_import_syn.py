# prepare data for snntorch
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import pickle
from concurrent.futures import ProcessPoolExecutor
from sklearn.utils import shuffle
import random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import snntorch as snn
import torch
import re



sub_dirs = ['1_4', '2_4', '3_4', '4_4', '5_4', '7_8']

CORE_COUNT : int = int(cpu_count()-1) 
num_features = 1  # onset_strength and BPM # pnly onset_strength # surrogate + onset
DIRTY = True
onset_padding = 20
FILES_TO_LOAD = 10
SR = int(16000)

print(f"Number of cores used: {CORE_COUNT}")


def parallel_data_loader(directories):
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(parallel_load_and_preprocess, directories), total=len(directories)))
    return results

def load_and_preprocess_data_subdir(args):
    directory, subdir = args
    data = []
    labels = []
    val_bpm = []
    
    # Only load up to 20 files per subdirectory
    files_to_load = os.listdir(os.path.join(directory, subdir))[:FILES_TO_LOAD]

    
    for file in files_to_load:
        file_path = os.path.join(directory, subdir, file)
        instrument, bpm, _, _ = extract_bpm_and_instrument(file_path)
        processed_data = preprocess_audio(file_path)
        label = int(bpm)
        for segment, bpm_librosa in processed_data:
            data.append(segment)
            labels.append(label)  # use the ground truth bpm as the label
            val_bpm.append(bpm_librosa)
    
    return data, labels, val_bpm


def extract_bpm_and_instrument(file_path):
    # Using \d+ to match one or more digits and [\d.]+ to match a float or integer pattern for noise.
    match = re.search(r"instrument_(\d+)_bpm_(\d+)_rotation_\d+_duration_(\d+)_noise_([\d.]+)", file_path)
    if match:
        instrument = match.group(1)
        bpm = match.group(2)
        duration = match.group(3)
        noise = match.group(4)
        return instrument, bpm, duration, noise
    return None, None, None, None

def parallel_load_and_preprocess(directory):
    # Create a pool of processes
    pool = Pool(CORE_COUNT)

    # Create a list of tasks
    tasks = [(directory, time_sig) for time_sig in sub_dirs]

    # Use imap_unordered to distribute the work among the processes
    results = list(tqdm(pool.imap_unordered(load_and_preprocess_data_subdir, tasks), total=len(tasks), mininterval=0.01))

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    # Combine results
    combined_data = []
    combined_labels = []
    combined_bpm = []
    
    for data, labels, val_bpm in results:
        combined_data.extend(data)
        combined_labels.extend(labels)
        combined_bpm.extend(val_bpm)
        
    
    return combined_data, combined_labels, combined_bpm


def sliding_window(data, window_size, step_size):
    """
    Split the data into overlapping windows. 
    For data that's shorter than the window size, just return the entire data as a single window.

    :param data: The data to be split into windows.
    :param window_size: The size of each window.
    :param step_size: The distance between the start points of consecutive windows.
    :return: A list of windows.
    """
    if len(data) <= window_size:
        return []
    
    num_windows = (len(data) - window_size) // step_size + 1

    if num_windows <= 0:
        return []

    windows = [data[i * step_size:i * step_size + window_size] for i in range(num_windows)]
    
    if len(windows[-1]) != window_size:
        windows.pop()
    
    return windows

   
def is_silent(segment, sr, threshold=0.01):
    """Check if the segment is silent based on its RMS energy."""
    rms_value = np.sqrt(np.mean(segment**2))
    return rms_value < threshold

def has_low_onset(segment, sr, onset_threshold=0.5):
    """Check if the segment has low onset strength."""
    onset_strengths = librosa.onset.onset_strength(y=segment, sr=sr)
    mean_onset_strength = np.mean(onset_strengths)
    return mean_onset_strength < onset_threshold

def preprocess_audio(file_path):
    # y, sr = librosa.load(file_path, sr=22050)  # setting sr ensures all files are resampled to this rate
    y, sr = librosa.load(file_path, sr=SR)  # setting sr ensures all files are resampled to this rate
    
    window_size = 6 * SR  # 6 seconds multiplied by the sampling rate
    step_size = window_size // 2  # 50% overlap
    segments = sliding_window(y, window_size, step_size)
    
    
    
    segment_features = []
    for segment in segments:
        if is_silent(segment, sr):
            print("Skipping this data segment because it's silent")
            continue
        
        # Skip segments with low onsets (nothing interesting going on)
        if has_low_onset(segment, sr):
            print("Skipping this data segment because it's spectral flux is too low")
            continue
        # Extracting Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr)
            # Extracting BPM (Tempo)
        tempo, _ = librosa.beat.beat_track(y=segment, sr=sr)
        
        # Get the Mel frequency region of interest
        if True: 
            mel_freqs = librosa.core.mel_frequencies(n_mels=mel_spectrogram.shape[0], fmin=0, fmax=sr/2)
            
            # Identify indices corresponding to 20Hz and 4kHz
            idx_start = np.where(mel_freqs >= 20)[0][0]
            idx_end = np.where(mel_freqs <= 4000)[0][-1]
            
            # Slice the Mel spectrogram to retain only the desired bands
            mel_spectrogram = mel_spectrogram[idx_start:idx_end+1, :]
        max_val = np.max(mel_spectrogram)
        min_val = np.min(mel_spectrogram)
        # Adjusting Mel spectrogram length if necessary (similar to onset_strength_adjusted)
        # mel_spectrogram_adjusted =  adjust_fixed_length(mel_spectrogram, fixed_timesteps, 1) # adjust as needed
        if max_val - min_val == 0:  # Check if denominator is zero
            print("Skipping this data point because the Mel spectrogram is all zeros")
            continue  # Skip this data point and move on to the next segment
        else:
            mel_spectrogram_normalized = (mel_spectrogram - np.min(mel_spectrogram)) / (np.max(mel_spectrogram) - np.min(mel_spectrogram))
        
        combined_features = np.vstack(mel_spectrogram_normalized), tempo
        
        segment_features.append(combined_features)
        

    return segment_features

def set_files_to_load(files_to_load, sr):
    global FILES_TO_LOAD, SR
    FILES_TO_LOAD = files_to_load
    SR = sr
    

class CustomAudioDataset(Dataset):
    def __init__(self, data, groundtruth, bpm_librosa):
        self.data = data
        self.groundtruth = groundtruth
        self.bpm_librosa = bpm_librosa

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        groundtruth = self.groundtruth[idx]
        bpm_librosa = self.bpm_librosa[idx]

        # Convert to PyTorch tensors
        sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)  # Add a channel dimension
        groundtruth = torch.tensor(groundtruth, dtype=torch.long)
        bpm_librosa = torch.tensor(bpm_librosa, dtype=torch.float32)

        return sample, groundtruth, bpm_librosa
    

def import_audio_get_loader(batch_size = 32, only_dataset = False, train_data_ratio = 0.8, files_to_load = None, sr = 12000):
    
    if files_to_load:
        set_files_to_load(files_to_load, sr)
    
        # checking shapes
    training_data_path = 'training_data_dirty_bpm' 
    validation_data_path = 'validation_data_dirty_bpm'   # validation_data_path = 'validation_data_dirty_bpm' if DIRTY else 'validation_data_clean'

    # print(fixed_timesteps)
    print("Loading and preprocessing training data...")
    directories = [training_data_path, validation_data_path]
    # training_data_results, validation_data_results = parallel_data_loader(directories)
    training_data_results, validation_data_results = parallel_data_loader(directories)
    
    training_data, training_labels, training_bpms = training_data_results
    validation_data, validation_labels, validation_bpms = validation_data_results
    # validation_data, validation_labels, validation_bpms, validation_genres = validation_data_results
    print("\nDone with preprocessing!")
    
    print(f"preparing data loaders with batch size {batch_size}")
    print(f"length of training data: {len(training_data)}")
    # print(f"sample data containments: {training_data[0]} ")

    
    train_dataset = CustomAudioDataset(training_data+validation_data, training_labels+validation_labels, training_bpms+validation_bpms)

    from torch.utils.data import random_split

    train_len = int(train_data_ratio * len(train_dataset))  # 80% for training
    val_len = len(train_dataset) - train_len  # 20% for validation
    
    train_dataset, val_dataset = random_split(train_dataset, [train_len, val_len])
    
    if only_dataset:
        return train_dataset, val_dataset, training_data[0].shape


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, val_loader, training_data[0].shape
