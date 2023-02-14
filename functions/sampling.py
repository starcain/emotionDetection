#packages imports..
import os
import random
import numpy as np
import pandas as pd
from scipy.io import loadmat

#functions imports..
if __name__ == "__main__":
    from pathlabelchannel import remove_channels
else:
    from functions.pathlabelchannel import remove_channels


def generate_eeg_samples(mat_file_path: str, label: list, channel: list, drop_index: list) -> tuple:
    mat_dict = loadmat(mat_file_path)
    keys = list(mat_dict.keys())
    
    final_sample = []
    final_hotkey = []
    
    for vid in range(3, 27):
        eeg_data = mat_dict[keys[vid]]

        drop_channels = remove_channels(drop_index)

        df = pd.DataFrame(eeg_data, index=channel)
        channel_no = 62 - len(drop_index)
        df = df.drop(index=drop_channels)

        eeg_data = df.to_numpy()

        sample_no = np.shape(eeg_data)[1] // 1000

        sample = np.zeros((sample_no, channel_no, 1000))

        for i in range(sample_no):
            for c in range(channel_no):
                sample[i, c, :] = eeg_data[c, i * 1000:(i + 1) * 1000]

        hotkey = np.zeros((sample_no, 4))
        hotkey[:, label[vid]] = 1

        if vid == 3:
            final_sample = sample
            final_hotkey = hotkey
        else:
            final_sample = np.concatenate((final_sample, sample), axis=0)
            final_hotkey = np.concatenate((final_hotkey, hotkey), axis=0)
        
    return final_sample, final_hotkey


def generate_batched_samples_from_directory(path: str, label: list, channel: list, drop_index: list = []) -> tuple:
    filenames = os.listdir(path)
    random.shuffle(filenames)
    print(filenames)
    
    finalsample, finalhotkey = np.zeros(0), np.zeros(0)

    for i in filenames:
        print(i)
        sample, hotkey = generate_eeg_samples(os.path.join(path, i), label, channel, drop_index)

        if not np.size(finalsample):
            finalsample = sample
            finalhotkey = hotkey
        else:
            sample = np.concatenate((finalsample, sample), axis=0)
            hotkey = np.concatenate((finalhotkey, hotkey), axis=0)
            
    return finalsample, finalhotkey
