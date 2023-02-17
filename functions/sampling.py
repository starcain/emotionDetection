#Import necessary modules..
from os import listdir
from os.path import join
from random import shuffle
from pandas import DataFrame
from scipy.io import loadmat
import numpy as np

if __name__ == "__main__":
    from pathlabelchannel import remove_channels
else:
    from functions.pathlabelchannel import remove_channels


# This function generates EEG samples and their corresponding hotkeys from a .mat file.
# The function takes the file path of the .mat file, labels, channels, and a list of indices to be dropped from the EEG data.
def generate_eeg_samples(mat_file_path: str, label: list, channel: list, drop_index: list) -> tuple:
    # Load the .mat file and get the keys of the dictionary.
    mat_dict = loadmat(mat_file_path)
    keys = list(mat_dict.keys())
    
    # Initialize empty numpy arrays to store the final samples and hotkeys.
    final_sample = []
    final_hotkey = []
    
    # Loop over each video (starting from the 4th key, since the first 3 are not relevant).
    for vid in range(3, 27):
        # Get the EEG data from the dictionary using the key.
        eeg_data = mat_dict[keys[vid]]

        # Remove the channels that are not required.
        drop_channels = remove_channels(drop_index)

        # Convert EEG data to a pandas DataFrame for easier manipulation.
        df = DataFrame(eeg_data, index=channel)

        # Remove the channels that are not required.
        channel_no = 62 - len(drop_index)
        df = df.drop(index=drop_channels)

        # Convert the DataFrame back to a numpy array.
        eeg_data = df.to_numpy()

        # Divide the data into 1000-sample segments to create individual samples.
        sample_no = np.shape(eeg_data)[1] // 1000
        sample = np.zeros((sample_no, channel_no, 1000))
        for i in range(sample_no):
            for c in range(channel_no):
                sample[i, c, :] = eeg_data[c, i * 1000:(i + 1) * 1000]

        # Create the corresponding hotkeys for each sample.
        hotkey = np.zeros((sample_no, 4))
        hotkey[:, label[vid]] = 1

        # Concatenate the samples and hotkeys to create the final output.
        if vid == 3:
            final_sample = sample
            final_hotkey = hotkey
        else:
            final_sample = np.concatenate((final_sample, sample), axis=0)
            final_hotkey = np.concatenate((final_hotkey, hotkey), axis=0)
        
    # Return the final samples and hotkeys as a tuple.
    return final_sample, final_hotkey


# This function generates batched EEG samples and their corresponding labels from a directory of EEG data files.
# The function takes the path of the directory, labels, channels, and a list of indices to be dropped from the EEG data.
def generate_batched_samples_from_directory(path: str, label: list, channel: list, drop_index: list = []) -> tuple:
    # Get a list of all file names in the directory and shuffle it randomly.
    filenames = listdir(path)
    shuffle(filenames)
    print(filenames)  # Print the shuffled file names.
    
    # Create empty numpy arrays for storing final samples and their hot keys.
    finalsample, finalhotkey = np.zeros(0), np.zeros(0)

    # Loop through all file names.
    for i in filenames:
        print(i)  # Print the current file name.
        # Generate EEG samples and their corresponding hot keys from the current file.
        sample, hotkey = generate_eeg_samples(join(path, i), label, channel, drop_index)

        # If finalsample is empty, set it to the current sample, otherwise concatenate the current sample to finalsample.
        if not np.size(finalsample):
            finalsample = sample
            finalhotkey = hotkey
        else:
            finalsample = np.concatenate((finalsample, sample), axis=0)
            finalhotkey = np.concatenate((finalhotkey, hotkey), axis=0)
    
    # Return the final samples and their corresponding hot keys as a tuple.
    return finalsample, finalhotkey

