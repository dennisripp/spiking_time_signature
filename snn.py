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
## Preprocessing files
fixed_timesteps = 1001
PLOT = True

def normalize_data(data):
    """Normalisiert eine Liste von Werten zwischen 0 und 1."""
    min_val = min(data)
    max_val = max(data)
    return [(val - min_val) / (max_val - min_val) for val in data]

def plot_monitors(input_data, input_monitor, hidden_monitor, output_monitor, output_spike_monitor):
    plt.figure(figsize=(15, 12))
    t_offset = np.min(input_monitor.t/ms)
    rows = 4
    # Plot for input layer
    plt.subplot(rows, 1, 1)
    plt.title('Input Layer')
    plt.plot(input_monitor.t/ms, input_monitor.v[0], label='Neuron 0')
    plt.plot(input_monitor.t/ms, input_monitor.v[1], label='Neuron 1')
    plt.plot(input_monitor.t/ms, input_monitor.v[2], label='Neuron 2')
    plt.plot(input_monitor.t/ms, input_monitor.v[3], label='Neuron 3')
    # ... repeat for other neurons if you want
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential')
    plt.legend()

    # Plot for hidden layer
    plt.subplot(rows, 1, 2)
    plt.title('Hidden Layer')
    plt.plot(hidden_monitor.t/ms, hidden_monitor.v[0], label='Neuron 0')
    plt.plot(hidden_monitor.t/ms, hidden_monitor.v[1], label='Neuron 1')
    plt.plot(hidden_monitor.t/ms, hidden_monitor.v[2], label='Neuron 2')
    plt.plot(hidden_monitor.t/ms, hidden_monitor.v[3], label='Neuron 3')
    plt.legend()
    # ... repeat for other neurons if you want
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential')

    # Plot for output layer
    plt.subplot(rows, 1, 3)
    plt.title('Output Layer')
    plt.plot(output_monitor.t/ms, output_monitor.v[0], label='Neuron 0')
    plt.plot(output_monitor.t/ms, output_monitor.v[1], label='Neuron 1')
    plt.plot(output_monitor.t/ms, output_monitor.v[2], label='Neuron 2')
    plt.plot(output_monitor.t/ms, output_monitor.v[3], label='Neuron 3')
    # ... repeat for other neurons if you want
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential')

    # Plot for output spikes
    plt.subplot(rows, 1, 4)
    plt.title('Input Layer')
    plt.plot(input_data, label='Input Data')
    # Add output spikes (assuming time is in ms)
    for idx in range(len(output_spike_monitor.count)):
        spike_times = output_spike_monitor.t[output_spike_monitor.i == idx] / ms
        plt.plot(spike_times - t_offset, [1] * len(spike_times), '.', label=f'Output Neuron {idx} Spike')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude / Spikes')
    plt.legend()

    plt.tight_layout()
    plt.show()

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


# Process the audio file into desired features
# Process the audio file into desired features
def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)  # setting sr ensures all files are resampled to this rate
    
    # Extracting onset strength
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
    
    # Extracting tempogram
    tempogram = librosa.feature.tempogram(onset_envelope=onset_strength, sr=sr)
    
    # Adjust the time axis of each feature to fixed_timesteps
    onset_strength = adjust_fixed_length(onset_strength, fixed_timesteps)
    tempogram = adjust_fixed_length(tempogram, fixed_timesteps)

    # Stacking features horizontally
    combined_features = np.vstack(onset_strength)
    
    # Normalize to range [0, 1]
    encoded_features = (combined_features - np.min(combined_features)) / (np.max(combined_features) - np.min(combined_features))
    return encoded_features


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

# Shuffle training data and labels
training_data, training_labels = shuffle(training_data, training_labels)

# sample_index = np.random.randint(0, len(training_data))

# for i in range(0, len(training_data)):
#     plt.plot(training_data[i])
#     plt.title(f"Random Sample from Normalized Data {training_labels[i]}")
#     plt.show()


## 2. Setting up the SNN:

start_scope()

n_input = len(training_data[0]) * len(training_data[0][0])

print(f"Number of input neurons: {n_input}")

n_hidden = 16  # Arbitrary
n_output = 4  # Four time signatures
# eligibility_trace_decay = 0.9  # Adjust based on desired behavior

##########################################################################
# Define LIF model
dropout_rate = 0.2
tau_m = 200*ms
v_rest = 0
v_thresh = 0.1  # v_threshold
v_reset = 0  # v_reset
r_m = 1.0  # membrane resistance
dt = 10*ms  # time step
eqs = '''
dv/dt = (-v + v_rest + r_m * I) / tau_m : 1
I : 1
'''

input_layer = NeuronGroup(n_input, eqs, threshold='v>'+str(v_thresh), reset='v=0', method='linear')
hidden_layer = NeuronGroup(n_hidden, eqs, threshold='v>'+str(v_thresh), reset='v=0', method='linear')
output_layer = NeuronGroup(n_output, eqs, threshold='v>'+str(v_thresh), reset='v=0', method='linear')

### 3. Training the SNN:

# # Define STDP
# tau_pre = 4000*ms  # Decreased slightly for stronger potentiation
# tau_post = 4500*ms  # Increased slightly for milder depression
# dropout_rate = 0.2

# A_pre = 0.0003  # Slightly reduce the potentiation
# A_post = -A_pre * 1.4  # Make depression stronger than potentiation
# delta_t = 10*ms  # the maximum time difference for STDP 
# w_max = 1.0
# w_min = 0.0
# decay_rate = 0.0033 #0.0005  # Experiment with this rate
# stdp_eqs = '''
# w : 1
# dpre/dt = -pre / tau_pre : 1 (event-driven)
# dpost/dt = -post / tau_post : 1 (event-driven)
# '''
# on_pre_eqs = '''
# I += w
# pre += A_pre
# w = clip(w + post - decay_rate, w_min, w_max)
# '''

# on_post_eqs = '''
# post += A_post
# w = clip(w + pre - decay_rate, w_min, w_max)
# '''
##########################################################################




# # Define LIF model
# tau = 300*ms
# v_thresh = 0.8
# eqs = '''
# dv/dt = (I + rest_potential - v)/tau : 1
# I : 1
# rest_potential = -0.1 : 1  # You can experiment with this value
# '''
# alpha = 0.001  # Regularization strength
# eqs += '''
# reg_penalty : 1
# '''
# eqs += '''
# e_trace : 1
# '''


# input_layer = NeuronGroup(n_input, eqs, threshold='v>'+str(v_thresh), reset='v=0', method='linear')
# hidden_layer = NeuronGroup(n_hidden, eqs, threshold='v>'+str(v_thresh), reset='v=0', method='linear')
# output_layer = NeuronGroup(n_output, eqs, threshold='v>'+str(v_thresh), reset='v=0', method='linear')

# ### 3. Training the SNN:

# # Define STDP
# dropout_rate = 0.2
tau_pre = 100*ms  # Decreased slightly for stronger potentiation
tau_post = 100*ms  # Increased slightly for milder depression
A_pre = 0.02  # Slightly reduce the potentiation
A_post = -A_pre * 1.2  # Make depression stronger than potentiation
delta_t = 20*ms  # the maximum time difference for STDP 
w_max = 1.0
w_min = 0.0
decay_rate =  0#0.0033 #0.0005  # Experiment with this rate
stdp_eqs = '''
w : 1
dpre/dt = -pre / tau_pre : 1 (event-driven)
dpost/dt = -post / tau_post : 1 (event-driven)
'''
on_pre_eqs = '''
I += w
pre += A_pre
w = clip(w + post - decay_rate, w_min, w_max)
'''

on_post_eqs = '''
post += A_post
w = clip(w + pre - decay_rate, w_min, w_max)
'''
# on_pre_eqs += '''
# reg_penalty += alpha * w  # Adjust this based on desired behavior
# '''
# on_pre_eqs += '''
# e_trace = e_trace * eligibility_trace_decay + w
# '''
# p = 1.0
# w = '0.5'
# w = 'rand()'

# synapses_input_hidden = Synapses(input_layer, hidden_layer, 'w : 1')
# synapses_input_hidden.connect(p=p)
# synapses_input_hidden.w = 'rand()'

# synapses_hidden_output = Synapses(hidden_layer, output_layer, 'w : 1')
# synapses_hidden_output.connect(p=p)
# synapses_hidden_output.w = 'rand()'

# Define and connect the synapses with STDP
p = 1.0
w = '0.5'
synapses_input_hidden = Synapses(input_layer, hidden_layer, model=stdp_eqs, on_pre=on_pre_eqs, on_post=on_post_eqs)
synapses_input_hidden.connect(p=p)
synapses_input_hidden.w = w

synapses_hidden_output = Synapses(hidden_layer, output_layer, model=stdp_eqs, on_pre=on_pre_eqs, on_post=on_post_eqs)
synapses_hidden_output.connect(p=p)
synapses_hidden_output.w = w

# with open('snn_model_state.pkl', 'rb') as f:
#     loaded_model_state = pickle.load(f)

# input_layer.set_states(loaded_model_state['input_layer'])
# hidden_layer.set_states(loaded_model_state['hidden_layer'])
# output_layer.set_states(loaded_model_state['output_layer'])
# synapses_input_hidden.set_states(loaded_model_state['synapses_input_hidden'])
# synapses_hidden_output.set_states(loaded_model_state['synapses_hidden_output'])

# Before training

total_samples = len(training_data)
training_accuracies = []

# Constants

print("Training network...")
epoch_cnt = 1
TIME_ALL = n_input * ms
TIME_STEP = TIME_ALL / 10

# In your training loop
for data_example, label in zip(training_data, training_labels):
    # Before setting the input current, use the Poisson encoding
    # poisson_encoded_example = poisson_spike_encoding(data_example, 1*second, TIME_STEP)
    # input_layer.I = poisson_encoded_example.flatten()
    input_layer.I = data_example.flatten()
    # Create a new SpikeMonitor for each iteration
    output_spike_monitor = SpikeMonitor(output_layer)
    input_monitor = StateMonitor(input_layer, 'v', record=True)
    hidden_monitor = StateMonitor(hidden_layer, 'v', record=True)
    output_monitor = StateMonitor(output_layer, 'v', record=True)
    
    
    # Now, instead of running the entire 1 second at once, break it into TIME_STEP increments.
    with tqdm(total=int((TIME_ALL)/TIME_STEP), leave=False, desc=f"Training [{epoch_cnt}/{len(training_data)}] completed") as pbar:
        for _ in range(int((TIME_ALL)/TIME_STEP)):
            run(TIME_STEP)
            pbar.update(1)   
    epoch_cnt += 1
      
    # Log spikes for current training sample (Optional)
    print(f"Spikes in the current sample: {output_spike_monitor.count[:]}")
    
    # Inside the training loop...
    # predicted_label = np.argmax([np.sum(output_layer.v == i) for i in range(4)])
    predicted_label = np.argmax(output_spike_monitor.count[:])
    if predicted_label == label:
        training_accuracies.append(1)
    else:
        training_accuracies.append(0)
        
    # active_neurons = np.random.binomial(1, 1 - dropout_rate, size=n_hidden)
    # hidden_layer.active = 'bool(active_neurons[_i])'  # _i refers to neuron index


        
    # Plot input data
    if PLOT: 
        all_empty = all(count == 0 for count in output_spike_monitor.count[:])
        if all_empty: 
            plot_monitors(data_example, input_monitor, hidden_monitor, output_monitor, output_spike_monitor)

    # Reset the state of the network
    input_layer.v = 0
    hidden_layer.v = 0
    output_layer.v = 0
    synapses_input_hidden.pre = 0
    synapses_input_hidden.post = 0
    synapses_hidden_output.pre = 0
    synapses_hidden_output.post = 0

      

print("\nTraining complete!")
accuracy_train = np.mean(training_accuracies)
print(f"Training Accuracy: {accuracy_train*100:.2f}%")
plt.figure()
plt.plot(np.cumsum(training_accuracies) / (np.arange(len(training_accuracies)) + 1))
plt.title(f'Training Accuracy Over Time {accuracy_train*100:.2f}%')
plt.xlabel('Training Sample')
plt.ylabel('Cumulative Accuracy')
plt.savefig('training_accuracy_curve.png')
plt.close()

# After training
model_state = {
    'input_layer': input_layer.get_states(),
    'hidden_layer': hidden_layer.get_states(),
    'output_layer': output_layer.get_states(),
    'synapses_input_hidden': synapses_input_hidden.get_states(),
    'synapses_hidden_output': synapses_hidden_output.get_states()
}

with open('snn_model_state.pkl', 'wb') as f:
    pickle.dump(model_state, f)


# After training, plotting spikes
plt.figure()
plt.plot(output_spike_monitor.t/ms, output_spike_monitor.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.savefig('spike_monitor_plot.png')  # Save plot as PNG
plt.close()  # Close the plot to free up memory


# Predict on training data and compute accuracy
print("\nPredict on training data and compute accuracy")
epoch_cnt = 1
predictions_train = []
correct_count_train = 0

# for data_example, true_label in zip(training_data, training_labels):
#     input_layer.I = data_example.flatten()
    
#         # Create a new SpikeMonitor for each iteration
#     output_spike_monitor = SpikeMonitor(output_layer)
      

#     # Now, instead of running the entire 1 second at once, break it into TIME_STEP increments.
#     with tqdm(total=int((1*second)/TIME_STEP), leave=False, desc=f"Predict [{epoch_cnt}/{len(training_data)}] completed") as pbar:
#         for _ in range(int((1*second)/TIME_STEP)):
#             run(TIME_STEP)
#             pbar.update(1)   
            
#     epoch_cnt += 1    
#         # Log spikes for current training sample (Optional)
#     print(f"Spikes in the current sample: {output_spike_monitor.count[:]}")
#     # predicted_label = np.argmax([np.sum(output_layer.v == i) for i in range(4)])
#     predicted_label = np.argmax(output_spike_monitor.count[:])
#     predictions_train.append(predicted_label)
#     if predicted_label == true_label:
#         correct_count_train += 1
#         # Reset the state of the network


accuracy_train = correct_count_train / len(training_labels)
print(f"Training Accuracy: {accuracy_train*100:.2f}%")


##############################################################################################
# Before starting validation, store the current weights to ensure they don't change
original_weights_input_hidden = synapses_input_hidden.w[:]
original_weights_hidden_output = synapses_hidden_output.w[:]

# Predict on validation data and compute accuracy
print("\nPredict on validation data and compute accuracy")
epoch_cnt = 1
predictions_val = []
correct_count_val = 0

# Disable STDP
synapses_input_hidden.active = False
synapses_hidden_output.active = False

for data_example, true_label in zip(validation_data, validation_labels):
    input_layer.I = data_example.flatten()
    
    output_spike_monitor = SpikeMonitor(output_layer)

    # Now, instead of running the entire 1 second at once, break it into TIME_STEP increments.
    with tqdm(total=int((1*second)/TIME_STEP), leave=False, desc=f"Validate [{epoch_cnt}/{len(validation_data)}] completed") as pbar:
        for _ in range(int((1*second)/TIME_STEP)):
            run(TIME_STEP)
            pbar.update(1)   
    epoch_cnt += 1
        # Log spikes for current training sample (Optional)
    print(f"Spikes in the current sample: {output_spike_monitor.count[:]}", end='\r')
    # Use the spike monitor approach for predictions
    predicted_label = np.argmax(output_spike_monitor.count[:])
    predictions_val.append(predicted_label)
    if predicted_label == true_label:
        correct_count_val += 1
    
    # Reset the state of the network
    input_layer.v = 0
    hidden_layer.v = 0
    output_layer.v = 0

# Re-enable STDP after validation
synapses_input_hidden.active = True
synapses_hidden_output.active = True

# Ensure weights remain the same after validation
assert np.array_equal(synapses_input_hidden.w[:], original_weights_input_hidden)
assert np.array_equal(synapses_hidden_output.w[:], original_weights_hidden_output)
##############################################################################################

accuracy_val = correct_count_val / len(validation_labels)
print(f"Validation Accuracy: {accuracy_val*100:.2f}%")

# Generate confusion matrices
confusion_matrix_train = np.zeros((4, 4))
for true_label, predicted_label in zip(training_labels, predictions_train):
    confusion_matrix_train[true_label, predicted_label] += 1

confusion_matrix_val = np.zeros((4, 4))
for true_label, predicted_label in zip(validation_labels, predictions_val):
    confusion_matrix_val[true_label, predicted_label] += 1

# Plot confusion matrix
plt.imshow(confusion_matrix_train, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Training Data Confusion matrix')
plt.colorbar()
tick_marks = np.arange(4)
plt.xticks(tick_marks, ['1_4', '2_4', '3_4', '4_4'])
plt.yticks(tick_marks, ['1_4', '2_4', '3_4', '4_4'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix_train.png')  # Save plot as PNG
plt.close()  # Close the plot to free up memory

# Validation
plt.imshow(confusion_matrix_val, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Validation Data Confusion matrix')
plt.colorbar()
tick_marks = np.arange(4)
plt.xticks(tick_marks, ['1_4', '2_4', '3_4', '4_4'])
plt.yticks(tick_marks, ['1_4', '2_4', '3_4', '4_4'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix_val.png')  # Save plot as PNG
plt.close()  # Close the plot to free up memory