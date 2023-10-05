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
from scipy import interpolate
import scipy.ndimage

sub_dirs = os.listdir("ballroom/BallroomData")

CORE_COUNT : int = int(cpu_count()-1) 
num_features = 1  # onset_strength and BPM # pnly onset_strength # surrogate + onset
DIRTY = True
onset_padding = 20
FILES_TO_LOAD = 10
TIME_DURATION = 6
SR = int(16000)
AUGMENTED = False

print(f"Number of cores used: {CORE_COUNT}")

def get_length(file_path):
    y, sr = librosa.load(file_path, sr=SR)
    mfccs = librosa.feature.melspectrogram(y=y, sr=sr)
    return mfccs.shape[1]

def is_silent(segment, sr, threshold=0.01):
    """Check if the segment is silent based on its RMS energy."""
    rms_value = np.sqrt(np.mean(segment**2))
    return rms_value < threshold

def has_low_onset(segment, sr, onset_threshold=0.5):
    """Check if the segment has low onset strength."""
    onset_strengths = librosa.onset.onset_strength(y=segment, sr=sr)
    mean_onset_strength = np.mean(onset_strengths)
    return mean_onset_strength < onset_threshold

def scale_time_axis(mel_spectrogram, scale_factor):
    num_mels, num_frames = mel_spectrogram.shape
    new_num_frames = int(scale_factor * num_frames)
    x_old = np.linspace(0, num_frames - 1, num_frames)
    x_new = np.linspace(0, num_frames - 1, new_num_frames)
    
    interpolator = interpolate.interp1d(x_old, mel_spectrogram, axis=1, kind='linear')
    scaled_spectrogram = interpolator(x_new)
    
    return scaled_spectrogram

def get_bpm_from_ground_truth(audio_path):
    # Extrahieren Sie nur den Dateinamen aus dem vollständigen Dateipfad
    file_name = os.path.basename(audio_path)
    # Erstellen Sie den Pfad zur entsprechenden Ground Truth-Datei
    ground_truth_path = os.path.join("ballroom/ballroomGroundTruth", file_name.replace(".wav", ".bpm"))
    
    with open(ground_truth_path, 'r') as f:
        bpm = float(f.readline().strip())
    return bpm

def parallel_data_loader(directories):
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(parallel_load_and_preprocess, directories), total=len(directories)))
    return results

def load_and_preprocess_data_subdir(args):
    directory, subdir = args
    data = []
    labels = []
    val_bpm = []

    
    all_files = os.listdir(os.path.join(directory, subdir))
    
    # Filtern nach Dateierweiterung (z.B. .wav für Audio-Dateien)
    filtered_files = [f for f in all_files if os.path.splitext(f)[1] == ".wav"]
    
    files_to_load = filtered_files[:FILES_TO_LOAD]

    for file in files_to_load:
        file_path = os.path.join(directory, subdir, file)
        bpm_ground_truth = get_bpm_from_ground_truth(file_path)

        processed_data = preprocess_audio(file_path, bpm_ground_truth)
        for segment, bpm_librosa, ground_truth_new in processed_data:
            data.append(segment)
            labels.append(ground_truth_new)  # use the ground truth bpm as the label
            val_bpm.append(bpm_librosa)

    return data, labels, val_bpm


def parallel_load_and_preprocess(directory):
    # Create a pool of processes
    pool = Pool(CORE_COUNT)

    # Create a list of tasks
    tasks = [(directory, genre) for genre in sub_dirs]

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
    Split the data into overlapping windows. Discard the last window if it's not long enough.

    :param data: The data to be split into windows.
    :param window_size: The size of each window.
    :param step_size: The distance between the start points of consecutive windows.
    :return: A list of windows.
    """
    num_windows = (len(data) - window_size) // step_size + 1
    windows = [data[i * step_size:i * step_size + window_size] for i in range(num_windows)]
    
    # Check the last window's length
    if len(windows[-1]) != window_size:
        windows.pop()  # Remove the last window if it doesn't match the window_size
    
    return windows

def stretch_and_crop(y, groundtruth, window_size):
    import random
    # Load the audio file    
    scaling_factor2 = np.random.uniform(0.6, 1.49)  # Random scaling factor between 0.8 and 1.2
    
    scaling_factor = random.uniform(0.6, 1.49)
        
    new_groundtruth = int(groundtruth * scaling_factor)

    # Stretch or shrink the audio data
    audio = librosa.effects.time_stretch(y=y, rate=scaling_factor)
    
    # Ensure the audio length is at least window_size
    if len(audio) < window_size:
        padding = np.zeros(window_size - len(audio))
        audio = np.concatenate((audio, padding))
    elif len(audio) > window_size:
        # Randomly crop the audio to window_size
        start_idx = random.randint(0, len(audio) - window_size)
        audio = audio[start_idx:start_idx + window_size]

    # Compute the mel spectrogram
    tempo, _ = librosa.beat.beat_track(y=audio, sr=SR)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=SR)
    
    return mel_spec, new_groundtruth, tempo


def preprocess_audio(file_path, ground_truth):
    global AUGMENTED
    # y, sr = librosa.load(file_path, sr=22050)  # setting sr ensures all files are resampled to this rate
    y, sr = librosa.load(file_path, sr=SR)  # setting sr ensures all files are resampled to this rate
    
    window_size = TIME_DURATION * SR  # 6 seconds multiplied by the sampling rate
    step_size = window_size // 2  # 50% overlap
    segments = sliding_window(y, window_size, step_size)
    tempo = 0
    
    
    segment_features = []
    for segment in segments:
        if is_silent(segment, sr):
            print("Skipping this data segment because it's silent")
            continue
        
        # Skip segments with low onsets (nothing interesting going on)
        if has_low_onset(segment, sr):
            print("Skipping this data segment because it's spectral flux is too low")
            continue

        
        if AUGMENTED:
            # select random scaling factor
            augmented_mel, ground_truth_adjusted, tempo = stretch_and_crop(y, ground_truth, window_size)
            mel_spectrogram = augmented_mel
            ground_truth = ground_truth_adjusted
        else:
            # Extracting Mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr)
                # Extracting BPM (Tempo)
            tempo, _ = librosa.beat.beat_track(y=segment, sr=sr)
            
        
        # # Get the Mel frequency values
        if False: 
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
        
        combined_features = np.vstack(mel_spectrogram_normalized), tempo, ground_truth
        
        segment_features.append(combined_features)
        

    return segment_features


def count_files(directory):
    return sum([len(files) for _, _, files in os.walk(directory)])

def set_files_to_load(files_to_load, sr, augmented):
    global FILES_TO_LOAD, SR, AUGMENTED
    FILES_TO_LOAD = files_to_load
    SR = sr
    AUGMENTED = augmented
    


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
    

def import_audio_get_loader(batch_size = 32, only_dataset = False, train_data_ratio = 0.8, files_to_load = None, sr = 12000, augmented = False):
    
    if files_to_load:
        set_files_to_load(files_to_load, sr, augmented)
    
        # checking shapes
    training_data_path = 'ballroom/BallroomData'
    # validation_data_path = 'validation_data_dirty_bpm' if DIRTY else 'validation_data_clean'

    # print(fixed_timesteps)
    print("Loading and preprocessing training data...")
    directories = [training_data_path]
    # training_data_results, validation_data_results = parallel_data_loader(directories)
    training_data_results, = parallel_data_loader(directories)

    training_data, training_labels, training_bpms = training_data_results
    # validation_data, validation_labels, validation_bpms, validation_genres = validation_data_results
    print("\nDone with preprocessing!")
    
    print(f"preparing data loaders with batch size {batch_size}")
    print(f"length of training data: {len(training_data)}")
    # print(f"sample data containments: {training_data[0]} ")

    
    train_dataset = CustomAudioDataset(training_data, training_labels, training_bpms)

    from torch.utils.data import random_split

    train_len = int(train_data_ratio * len(train_dataset))  # 80% for training
    val_len = len(train_dataset) - train_len  # 20% for validation
    
    train_dataset, val_dataset = random_split(train_dataset, [train_len, val_len])
    
    if only_dataset:
        return train_dataset, val_dataset, training_data[0].shape

    # sample_batch = train_dataset
    # sample, groundtrush, bpm, genre = sample_batch[0]

    # print("groundtruth: ", groundtrush.cpu().numpy())
    # print("bpm_librosa: ", bpm.cpu().numpy())
    # print("genre: ", genre.cpu().numpy())

    # print(sample[0].shape)
    # test_dataset = CustomAudioDataset(validation_data, validation_labels, validation_bpms, validation_genres)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, val_loader, training_data[0].shape


# train_loader, _ = import_audio_get_loader(batch_size = 32, only_dataset = False, train_data_ratio = 0.8, files_to_load = 2)

# data, targets, librosa_bpm, _ = next(iter(train_loader))

# single_mel = data[0].squeeze(0)  # select the first mel-spec from the batch and remove the channel dimension

# # librosa.display.specshow(librosa.power_to_db(data, ref=np.max), y_axis='mel', x_axis='time')
# plt.imshow(librosa.power_to_db(single_mel, ref=np.max), origin='lower', aspect='auto')
# plt.colorbar(format='%+2.0f dB')
# plt.show()


    
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

