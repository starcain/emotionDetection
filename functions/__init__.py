import numpy as np

if __name__ == "__main__":
    from sampling import generate_batched_samples_from_directory
    from cnnReduce import train_cnn_reduce
    from pathlabelchannel import get_path_label_channel_by_index
else:
    from functions.sampling import generate_batched_samples_from_directory
    from functions.cnnReduce import train_cnn_reduce
    from functions.pathlabelchannel import get_path_label_channel_by_index


def cnnmodel(index : int = 0):
    drop = 51
    remainingChannels = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 
    47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61
    ]

    dropIndex = [drop, remainingChannels[index]]
    print(dropIndex)

    path, label, channel = get_path_label_channel_by_index(0)

    sample, hotkey = generate_batched_samples_from_directory(path, label, channel, dropIndex)
    print(np.shape(sample), np.shape(hotkey))

    model = train_cnn_reduce(sample, hotkey)