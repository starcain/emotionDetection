from scipy.io import loadmat
import numpy as np
from os import listdir
from pandas import DataFrame


def sampling61(matfilePath : str, label : list, channel : list, dropIndex : int):

    matDict = loadmat(matfilePath)
    keys = list(matDict.keys())

    # for key in keys[3:]:
        # print(key)
    finalsample = []
    finalhotkey = []
    
    for vid in range(3,27):
        eegData = matDict[keys[vid]]

        df = DataFrame(eegData, index = channel)
        df = df.drop(index = channel[dropIndex])

        eegData = df.to_numpy()

        sampleNo = np.shape(eegData)[1] // 1000

        sample = np.zeros((sampleNo, 61, 1000))

        for i in range(sampleNo):
            for c in range(61):
                sample[i, c, :] = eegData[c, i * 1000:(i + 1) * 1000]

        hotkey = np.zeros((sampleNo, 4))
        hotkey[:, label[vid]] = 1

        if vid == 3:  # for first video case (no concatenation required)
            # print("fun-IF")
            finalsample = sample
            finalhotkey = hotkey
        else:  # for remaining cases concatenating to final array
            # print("fun-ELSE")
            finalsample = np.concatenate((finalsample, sample), axis=0)
            finalhotkey = np.concatenate((finalhotkey, hotkey), axis=0)
        
        # break

    return finalsample, finalhotkey
 


def dirsample61(path : list, label : list, channel : list, dropIndex : int):
    filenames = listdir(path)
    sample, hotkey = np.zeros(0), np.zeros(0)

    for i in filenames:
        print(i)
        sample61, hotkey61 = sampling61(path + i, label, channel, dropIndex)

        if not np.size(sample):
            # print("main()-IF")
            sample = sample61
            hotkey = hotkey61
        else:
            # print("main()-ELSE")
            sample = np.concatenate((sample, sample61), axis=0)
            hotkey = np.concatenate((hotkey, hotkey61), axis=0)
            # break
    return sample, hotkey
