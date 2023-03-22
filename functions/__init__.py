import numpy as np
from gc import collect


from functions.sampling import generate_batched_samples_from_directory, generate_all_batched_samples
from functions.cnnOptimize import train_cnn
from functions.pathlabelchannel import get_path_label_channel_by_index, get_path_label_channel

path, label, channel = get_path_label_channel()

Channels = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61
    ]

drop = [
    5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19, 21, 24, 25, 26, 27, 28, 29, 
    30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47, 48, 49, 
    51, 53, 54, 55, 56, 57, 58, 59, 60, 61
    ]

remainingChannels = [ 
    0, 1, 2, 3, 4, 10, 13, 20, 22, 
    23, 32, 42, 43, 50, 52,
    ]

def model14(index : int = 0):
    dropIndex = drop.copy()
    dropIndex.append(Channels[index])
    dropIndex = list(set(dropIndex))

    if len(dropIndex)==len(drop)+1:
        print("INDEX: ", index)
        print("Used Channels: ",62-len(dropIndex))
        print(dropIndex)
        cnnmodel(dropIndex)
    else:
        print(index," :(")
    
    # Trigger garbage collection to free up memory.
    gc.collect()

def model14_all(index : int = 0, pathindex : list = [0, 1, 2]):
    dropIndex = drop.copy()
    dropIndex.append(Channels[index])
    dropIndex = list(set(dropIndex))

    if len(dropIndex)==len(drop)+1:
        print("INDEX: ", index)
        print("Path Index: ", pathindex)
        print("Used Channels: ",62-len(dropIndex))
        print(dropIndex)
        cnnmodel_all(dropIndex, pathindex)
    else:
        print(index," :(")
    
    # Trigger garbage collection to free up memory.
    collect()

def cnnmodel(dropChannels : list):
    sample, hotkey = generate_batched_samples_from_directory(path[0], label[0], channel, dropChannels)
    shape = np.shape(sample)
    print(shape)

    model = train_cnn(sample, hotkey, channel=shape[1])
    
    # Trigger garbage collection to free up memory.
    collect()

def cnnmodel_all(dropChannels : list, pathindex : list):
    sample, hotkey = generate_all_batched_samples(path, label, channel, pathindex, dropChannels)
    shape = np.shape(sample)
    print(shape)

    model = train_cnn(sample, hotkey, channel=shape[1])

    # Trigger garbage collection to free up memory.
    collect()
