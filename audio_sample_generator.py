"""multi-threaded sample generation system"""


import os
import random
import numpy as np
import soundfile as sf
import pretty_midi
from midi2audio import FluidSynth
import librosa
from multiprocessing import Pool
import tempfile  # Make sure to import this at the beginning of your script
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from collections import deque

soundfont_path = 'soundfonts/SoundFonts_GeneralUser GS v1.471.sf2'

dirty = True

# if not dirty:
SAMPLE_CNT = 1000
SR = int(44100/2)
DURATION = 12 # seconds
BPM = 120
MAX_BPM = 200
MIN_BPM = 80

CLEAR_INSTRUMENTS = [0, 112, 113, 114, 115, 116, 117, 118, 56, 30, 26, 17, 11]


def random_bpm(min_bpm=MIN_BPM, max_bpm=MAX_BPM):
    if dirty:    
        return random.randint(min_bpm, max_bpm)
    return BPM

def bpm_for_fixed_duration(time_signature, fixed_duration=12.0):
    beats_per_measure, base_note = time_signature
    new_bpm = random_bpm()
    
    # Estimate the number of beats that would fit into the fixed_duration for a standard BPM
    beats_in_fixed_duration = new_bpm * (fixed_duration / 60.0)
    
    # Calculate how many full cycles can fit into the fixed_duration
    full_cycles_in_duration = beats_in_fixed_duration / beats_per_measure
    
    # We want an integer number of full cycles to fit into the fixed_duration
    # Therefore, we'll round down to the nearest integer for full cycles
    full_cycles_in_duration = int(full_cycles_in_duration)
    
    # Now, we recalculate how many beats this represents
    total_beats = full_cycles_in_duration * beats_per_measure
    
    # Derive the new BPM such that the total_beats fit into the fixed_duration
    adjusted_bpm = total_beats / (fixed_duration / 60.0)
    
    return int(adjusted_bpm)

def random_bpm_based_on_duration(time_signature):

    return bpm_for_fixed_duration(time_signature)


# New formula to get note duration
def get_note_duration(bpm, base_note):
    whole_note_duration = 60.0 / bpm * 4  # Duration of a whole note in seconds
    return whole_note_duration / base_note


# CLEAR_INSTRUMENTS = [0, 76, 62, 65, 81]
#CLEAR_INSTRUMENTS = [0, 12, 19, 24, 31, 33, 40, 56, 58, 60, 61, 62, 64, 65, 68, 69, 71, 73, 76, 80, 81, 82, 86, 87, 89, 99, 105, 113]
def apply_dtw_to_sample(sample, target_length):
    # Assuming the sample and target_length are 1D numpy arrays
    target = np.zeros(target_length)  # Replace this with whatever your target sequence is
    distance, path = fastdtw(sample, target, dist=euclidean)
    
    new_sample = np.zeros(target_length)
    
    for sample_idx, target_idx in path:
        new_sample[target_idx] = sample[sample_idx]
        
    return new_sample


def generate_target_sequence(length, peaks, peak_value=1.0):
    """Generate a target sequence with peaks at given positions."""
    target = np.zeros(length)
    for peak in peaks:
        target[peak] = peak_value
    return target

def random_instrument():
    # if dirty:   
    #     return random.randint(0, 127)
    return random.choice(CLEAR_INSTRUMENTS)


# Update this function to accept an additional parameter for accent_profiles
def get_velocity_for_beat(note_num, time_signature, rotation, accent_profiles):
    profile = accent_profiles.get(time_signature, [80])
    rotated_profile = deque(profile)
    rotated_profile.rotate(rotation)
    rotated_profile = list(rotated_profile)
    position = note_num  % len(rotated_profile)
    return rotated_profile[position]

def get_accent_profiles(dirty):
    if dirty:
        return {
            (1, 4): [random.randint(115,127)],               # 1/4 time signature: single strong beat
            (2, 4): [random.randint(115,127), random.randint(80,95)],           # 2/4 time signature: strong-weak
            (3, 4): [random.randint(115,127), random.randint(100,110), random.randint(70,85)],       # 3/4 time signature: strong-medium-weak
            (4, 4): [random.randint(115,127), random.randint(100,110), random.randint(115,127), random.randint(70,85)],   # 4/4 time signature: strong-medium-strong-weak
            (5, 4): [random.randint(115,127), random.randint(115,127), random.randint(100,110), random.randint(100,110), random.randint(70,85)],  # 5/4 time signature: strong-strong-medium-medium-weak
            (7, 8): [random.randint(115,127), random.randint(115,127), random.randint(100,110), random.randint(100,110), random.randint(70,85), random.randint(70,85), random.randint(70,85)],  # 7/8 time signature: strong-strong-medium-medium-weak-weak-weak
        }
    else:    
        # # clean profile
        return  {
            (4, 4): [110, 40, 70, 40],   # 4/4 time signature
            (1, 4): [110],               # 1/4 time signature
            (2, 4): [110, 70],           # 2/4 time signature
            (3, 4): [110, 40, 70],       # 3/4 time signature
            (5, 4): [110, 40, 40, 70, 40],  # Odd time signature
            (7, 8): [110, 40, 40, 70, 40, 40, 40],  # Odd time signature
        }
    

def midi_to_audio(midi, soundfont_path):
    fs = FluidSynth(soundfont_path)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mid") as temp_midi_file:
        midi.write(temp_midi_file.name)
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_audio_file:
            fs.midi_to_audio(temp_midi_file.name, temp_audio_file.name)
            audio_data, _ = librosa.load(temp_audio_file.name, sr=SR)
    return audio_data


def add_noise(data, noise_factor=0.05):
    noise = np.random.randn(len(data))
    if dirty:
        return data + noise_factor * noise
    return data

# def random_instrument():
#     return random.randint(0, 127)

def random_duration(min_duration=5, max_duration=20):
    """Generate a random duration in seconds."""
    #return random.uniform(min_duration, max_duration)
    # if dirty:
    #     return random.uniform(min_duration, max_duration)
    return DURATION

def random_noise_factor():
    return random.uniform(0.01, 0.019)

## adding silence approach
def generate_midi_beat(bpm=120, time_signature=(4, 4), instrument=0, duration=10, rotation=1, accent_profiles=None):
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=instrument)
    midi.instruments.append(inst)
    
    beats_per_measure, base_note = time_signature
    
    # Calculate note duration
    note_duration = get_note_duration(bpm, base_note)
    
    total_notes = int(duration / note_duration)  # Total number of notes for the adjusted duration
    
    # Get accent profiles
    accent_profiles = get_accent_profiles(dirty)
    
    for note_num in range(total_notes):
        start = note_num * note_duration
        end = start + note_duration
        velocity = get_velocity_for_beat(note_num, time_signature, rotation, accent_profiles)
        note = pretty_midi.Note(velocity=velocity, pitch=60, start=start, end=end)
        inst.notes.append(note)
    
    return midi


def generate_random_sample(output_folder, time_signature=(4, 4)):
    bpm, instrument, duration = random_bpm_based_on_duration(time_signature), random_instrument(), DURATION
    
    # Generate a random rotation for the accent profile
    rotation = random.randint(0, time_signature[0] - 1)
    
    accent_profiles = get_accent_profiles(dirty)
    midi_data = generate_midi_beat(bpm=bpm, time_signature=time_signature, instrument=instrument, duration=duration, rotation=rotation, accent_profiles=accent_profiles)    
    audio_data = midi_to_audio(midi_data, soundfont_path)
    
    noise_factor = random_noise_factor()
    audio_with_noise = add_noise(audio_data, noise_factor=noise_factor)
    
    filename = f"instrument_{instrument}_bpm_{bpm}_rotation_{rotation}_duration_{round(duration, 2)}_noise_{round(noise_factor, 2)}_{random.randint(0, 9999999)}.wav"
    output_path = os.path.join(output_folder, f"{time_signature[0]}_{time_signature[1]}", filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio_with_noise.T, SR)



def generate_samples_for_type(data_type, time_signature, num_samples=500):
    for _ in range(num_samples):
        generate_random_sample(data_type, time_signature)
        
def generate_samples(num_samples=500):
    time_signatures = [(4, 4), (1, 4), (2, 4), (3, 4), (5, 4), (7, 8)]
    train_folder = 'training_data_clean' if not dirty else 'training_data_dirty_bpm'
    val_folder = 'validation_data_clean' if not dirty else 'validation_data_dirty_bpm'
    data_types = [train_folder, val_folder]

    if os.path.exists(train_folder):
        os.system(f"rm -rf {train_folder}")
    if os.path.exists(val_folder):
        os.system(f"rm -rf {val_folder}")    
    
    tasks = [(data_type, ts, num_samples) for data_type in data_types for ts in time_signatures]
    
    with Pool() as pool:
        pool.starmap(generate_samples_for_type, tasks)
        
# Example
generate_samples(SAMPLE_CNT)

# # Anzahl der möglichen Instrumente
# num_instruments = 13

# # Summe der Rotationen für alle Taktarten
# time_signatures = [(4, 4), (1, 4), (2, 4), (3, 4), (5, 4), (7, 8)]
# total_rotations = sum([ts[0] for ts in time_signatures])

# # Anzahl der BPMs
# num_bpms = 200 - MIN_BPM + 1

# # Anzahl der Taktarten
# num_time_signatures = len(time_signatures)

# # Gesamtzahl der möglichen Kombinationen
# total_combinations = num_instruments * total_rotations * num_bpms * num_time_signatures

# print(total_combinations)
# print(num_instruments)
