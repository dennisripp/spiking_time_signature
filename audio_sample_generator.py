import os
import random
import numpy as np
import soundfile as sf
import pretty_midi
from midi2audio import FluidSynth
import librosa
from multiprocessing import Pool
import tempfile  # Make sure to import this at the beginning of your script

soundfont_path = 'soundfonts/SoundFonts_GeneralUser GS v1.471.sf2'


def get_velocity_for_beat(note_num, time_signature):
    # Define accent profiles for different time signatures
    # For instance, for 4/4: first beat is strongly accented, third is mildly accented, and others are regular.
    accent_profiles = {
        (4, 4): [random.randint(110,127), random.randint(70,90), random.randint(90,110), random.randint(70,90)],   # 4/4 time signature
        (1, 4): [random.randint(110,127)],               # 1/4 time signature
        (2, 4): [random.randint(110,127), random.randint(70,90)],           # 2/4 time signature
        (3, 4): [random.randint(110,127), random.randint(70,90), random.randint(70,90)],       # 3/4 time signature
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
            audio_data, _ = librosa.load(temp_audio_file.name, sr=22050)
    return audio_data


def add_noise(data, noise_factor=0.05):
    noise = np.random.randn(len(data))
    return data + noise_factor * noise
   # return data

def random_bpm(min_bpm=60, max_bpm=180):
    return random.randint(min_bpm, max_bpm)

def random_instrument():
    return random.randint(0, 127)

def random_duration(min_duration=5, max_duration=20):
    """Generate a random duration in seconds."""
    #return random.uniform(min_duration, max_duration)
    return 10

def random_noise_factor():
    return random.uniform(0.01, 0.03)

def generate_random_sample(output_folder, time_signature=(4, 4)):
    bpm, instrument, duration = random_bpm(), random_instrument(), random_duration()
    midi_data = generate_midi_beat(bpm=bpm, time_signature=time_signature, instrument=instrument, duration=duration)
    audio_data = midi_to_audio(midi_data, soundfont_path)
    noise_factor = random_noise_factor()
    audio_with_noise = add_noise(audio_data, noise_factor=noise_factor)
    
    filename = f"instrument_{instrument}_bpm_{bpm}_duration_{round(duration, 2)}_noise_{round(noise_factor, 2)}_{random.randint(0, 99999)}.wav"
    output_path = os.path.join(output_folder, f"{time_signature[0]}_{time_signature[1]}", filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio_with_noise.T, 22050)

def generate_midi_beat(bpm=120, time_signature=(4, 4), instrument=0, duration=10):
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=instrument)
    midi.instruments.append(inst)
    
    note_duration = 60.0 / bpm  # Duration of each note in seconds
    beats_per_measure, base_note = time_signature
    total_notes = int(duration / note_duration)  # Total number of notes for the specified duration
    
    for note_num in range(total_notes):
        start, end = note_num * note_duration, (note_num + 1) * note_duration
        velocity = get_velocity_for_beat(note_num, time_signature)
        note = pretty_midi.Note(velocity=velocity, pitch=60, start=start, end=end)
        inst.notes.append(note)
    
    return midi

def generate_samples_for_type(data_type, time_signature, num_samples=500):
    for _ in range(num_samples):
        generate_random_sample(data_type, time_signature)
        
def generate_samples(num_samples=500):
    time_signatures = [(4, 4), (1, 4), (2, 4), (3, 4)]
    data_types = ['training_data', 'validation_data']
    
    tasks = [(data_type, ts, num_samples) for data_type in data_types for ts in time_signatures]
    
    with Pool() as pool:
        pool.starmap(generate_samples_for_type, tasks)
        
# Example
generate_samples(6)
