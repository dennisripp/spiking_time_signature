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

soundfont_path = 'soundfonts/SoundFonts_GeneralUser GS v1.471.sf2'

dirty = False
SAMPLE_CNT = 20
SR = int(44100/2)

CLEAR_INSTRUMENTS = [0]
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

# def add_silence_around_onsets(audio_data, samples_per_beat, silence_duration=0.1):
#     """
#     Adds a period of silence around each onset in the audio data.
#     """
#     silence_samples = int(SR * silence_duration)
#     new_audio_data = np.zeros(len(audio_data) + silence_samples * 2 * int(len(audio_data) / samples_per_beat))
    
#     for i in range(0, len(audio_data), samples_per_beat):
#         onset_data = audio_data[i : i + samples_per_beat]
#         onset_length = len(onset_data)
        
#         if onset_length < samples_per_beat:
#             onset_data = np.pad(onset_data, (0, samples_per_beat - onset_length), 'constant')  # Zero padding to fill the last slice

#         new_audio_data[i + silence_samples : i + silence_samples + samples_per_beat] = onset_data
        
#     return new_audio_data

def generate_target_sequence(length, peaks, peak_value=1.0):
    """Generate a target sequence with peaks at given positions."""
    target = np.zeros(length)
    for peak in peaks:
        target[peak] = peak_value
    return target

def random_instrument():
    if dirty:   
        return random.randint(0, 127)
    return random.choice(CLEAR_INSTRUMENTS)

def get_velocity_for_beat(note_num, time_signature):
    # Define accent profiles for different time signatures
    # For instance, for 4/4: first beat is strongly accented, third is mildly accented, and others are regular.
    
    accent_profiles = { }
    
    if dirty:
        accent_profiles = {
            (4, 4): [random.randint(110,127), random.randint(70,90), random.randint(90,110), random.randint(70,90)],   # 4/4 time signature
            (1, 4): [random.randint(110,127)],               # 1/4 time signature
            (2, 4): [random.randint(110,127), random.randint(70,90)],           # 2/4 time signature
            (3, 4): [random.randint(110,127), random.randint(70,90), random.randint(70,90)],       # 3/4 time signature
        }
    else:    
        # # clean profile
        accent_profiles = {
            (4, 4): [110, 40, 70, 40],   # 4/4 time signature
            (1, 4): [110],               # 1/4 time signature
            (2, 4): [110, 70],           # 2/4 time signature
            (3, 4): [110, 40, 70],       # 3/4 time signature
        }
    
    # Get the accent profile for the given time signature
    profile = accent_profiles.get(time_signature, [80])  # Default to regular beat
    
    # Use modulo to get the position of the beat within the measure
    position = note_num % len(profile)
    
    return profile[position]

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

def random_bpm(min_bpm=60, max_bpm=180):
    if dirty:    
        return random.randint(min_bpm, max_bpm)
    return 120

# def random_instrument():
#     return random.randint(0, 127)

def random_duration(min_duration=5, max_duration=20):
    """Generate a random duration in seconds."""
    #return random.uniform(min_duration, max_duration)
    return 10

def random_noise_factor():
    return random.uniform(0.01, 0.019)

# adding silence approach
# def generate_random_sample(output_folder, time_signature=(4, 4)):
#     bpm, instrument, duration = random_bpm(), random_instrument(), random_duration()
#     midi_data = generate_midi_beat(bpm=bpm, time_signature=time_signature, instrument=instrument, duration=duration)
#     audio_data = midi_to_audio(midi_data, soundfont_path)

#     # Calculate samples per beat, assuming constant tempo
#     samples_per_beat = int(SR * 60 / bpm)

#     # Add silence around onsets
#     audio_data_with_silence = add_silence_around_onsets(audio_data, samples_per_beat)
    
#     noise_factor = random_noise_factor()
#     audio_with_noise = add_noise(audio_data_with_silence, noise_factor=noise_factor)
    
#     filename = f"instrument_{instrument}_bpm_{bpm}_duration_{round(duration, 2)}_noise_{round(noise_factor, 2)}_{random.randint(0, 99999)}.wav"
#     output_path = os.path.join(output_folder, f"{time_signature[0]}_{time_signature[1]}", filename)
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     sf.write(output_path, audio_with_noise, SR)

# Dynamic Time Warping (DTW) version (expensive computationally)
# def generate_random_sample(output_folder, time_signature=(4, 4)):
#     bpm, instrument, duration = random_bpm(), random_instrument(), random_duration()
#     midi_data = generate_midi_beat(bpm=bpm, time_signature=time_signature, instrument=instrument, duration=duration)
#     audio_data = midi_to_audio(midi_data, soundfont_path)

#     # Calculate samples per beat, assuming constant tempo
#     samples_per_beat = int(SR * 60 / bpm)

#     # Generate list of positions for onsets
#     peak_positions = [i * samples_per_beat for i in range(int(duration * bpm / 60))]

#     # Generate the target sequence
#     target_sequence = generate_target_sequence(len(audio_data), peak_positions)

#     # Apply DTW to align audio data to the target sequence
#     audio_data_aligned = apply_dtw_to_sample(audio_data, len(target_sequence))
    
#     noise_factor = random_noise_factor()
#     audio_with_noise = add_noise(audio_data_aligned, noise_factor=noise_factor)
    
#     filename = f"instrument_{instrument}_bpm_{bpm}_duration_{round(duration, 2)}_noise_{round(noise_factor, 2)}_{random.randint(0, 99999)}.wav"
#     output_path = os.path.join(output_folder, f"{time_signature[0]}_{time_signature[1]}", filename)
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     sf.write(output_path, audio_with_noise, SR)


#     ordinary approach
# def generate_random_sample(output_folder, time_signature=(4, 4)):
#     bpm, instrument, duration = random_bpm(), random_instrument(), random_duration()
#     midi_data = generate_midi_beat(bpm=bpm, time_signature=time_signature, instrument=instrument, duration=duration)
#     audio_data = midi_to_audio(midi_data, soundfont_path)
#     noise_factor = random_noise_factor()
#     audio_with_noise = add_noise(audio_data, noise_factor=noise_factor)
    
#     filename = f"instrument_{instrument}_bpm_{bpm}_duration_{round(duration, 2)}_noise_{round(noise_factor, 2)}_{random.randint(0, 99999)}.wav"
#     output_path = os.path.join(output_folder, f"{time_signature[0]}_{time_signature[1]}", filename)
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     sf.write(output_path, audio_with_noise.T, SR)

# def generate_midi_beat(bpm=120, time_signature=(4, 4), instrument=0, duration=10):
#     midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
#     inst = pretty_midi.Instrument(program=instrument)
#     midi.instruments.append(inst)
    
#     note_duration = 60.0 / bpm  # Duration of each note in seconds
#     beats_per_measure, base_note = time_signature
#     total_notes = int(duration / note_duration)  # Total number of notes for the specified duration
    
#     for note_num in range(total_notes):
#         start, end = note_num * note_duration, (note_num + 1) * note_duration
#         velocity = get_velocity_for_beat(note_num, time_signature)
#         note = pretty_midi.Note(velocity=velocity, pitch=60, start=start, end=end)
#         inst.notes.append(note)
    
#     return midi

## adding silence approach
def generate_midi_beat(bpm=120, time_signature=(4, 4), instrument=0, duration=10, silence_factor=0.2):
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=instrument)
    midi.instruments.append(inst)
    
    note_duration = 60.0 / bpm  # Duration of each note in seconds
    silence_duration = note_duration * silence_factor  # Duration of silence in seconds
    
    beats_per_measure, base_note = time_signature
    total_notes = int(duration / (note_duration + silence_duration))  # Total number of notes for the specified duration
    
    for note_num in range(total_notes):
        start = note_num * (note_duration + silence_duration)
        end = start + note_duration  # End before the silence starts
        velocity = get_velocity_for_beat(note_num, time_signature)
        note = pretty_midi.Note(velocity=velocity, pitch=60, start=start, end=end)
        inst.notes.append(note)
    
    return midi

def generate_random_sample(output_folder, time_signature=(4, 4)):
    bpm, instrument, duration = random_bpm(), random_instrument(), random_duration()
    midi_data = generate_midi_beat(bpm=bpm, time_signature=time_signature, instrument=instrument, duration=duration)
    audio_data = midi_to_audio(midi_data, soundfont_path)
    
    noise_factor = random_noise_factor()
    audio_with_noise = add_noise(audio_data, noise_factor=noise_factor)
    
    filename = f"instrument_{instrument}_bpm_{bpm}_duration_{round(duration, 2)}_noise_{round(noise_factor, 2)}_{random.randint(0, 99999)}.wav"
    output_path = os.path.join(output_folder, f"{time_signature[0]}_{time_signature[1]}", filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio_with_noise.T, SR)


def generate_samples_for_type(data_type, time_signature, num_samples=500):
    for _ in range(num_samples):
        generate_random_sample(data_type, time_signature)
        
def generate_samples(num_samples=500):
    time_signatures = [(4, 4), (1, 4), (2, 4), (3, 4)]
    train_folder = 'training_data'
    val_folder = 'validation_data'
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
