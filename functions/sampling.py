from scipy.io import loadmat
import numpy as np
from os import listdir
from pandas import DataFrame

def sampling(matfile : str, hlabel : list):
    # takes a matfile, label list as input and return samples,hotkeys

    # loading the matfile dictionary data in dict
    matdict = loadmat(matfile)

    keys = []  # to store all the keys in dict

    for key in matdict:
        # appending every single key from the dict
        keys.append(key)

    # print(keys)
    # to check keys in file

    # blank sample, hotkey array defined
    finalsample = []
    finalhotkey = []

    for vid in range(3, 27):
        # reading all data of dictionary using keys, excluding '__header__', '__version__', '__globals__'
        data = matdict[keys[vid]]  # extracting data from dictionary using keys

        y = np.shape(data)[1]  # returns the shape of the data
        # print(x,y)

        sno = y // 1000  # no of sample obtained of 1000 width each
        # print(sno, end=',')
        #         hotkey[4]
        #         0123
        #         1000 0 neutral
        #         0100 1 sad
        #         0010 2 fear
        #         0001 3 happy

        # excluding the extra data according to number of sample

        sample = np.zeros((sno, 62, 1000))
        for i in range(sno):
            for c in range(62):
                sample[i, c, :] = data[c, i * 1000:(i + 1) * 1000]

        # reshaping of data(62,total) to sample(sno,62,1000)
        # sample = data.reshape(sno, 1000, 62)
        # print(np.shape(sample))

        hotkey = np.zeros((sno, 4))
        # blank hotkey array defined to store hotkey based on label
        hotkey[:, hlabel[vid]] = 1
        # 1000 0100 0010 0001 labelled accordingly from label

        if vid == 3:  # for first video case (no concatenation required)
            # print("fun-IF")
            finalsample = sample
            finalhotkey = hotkey
        else:  # for remaining cases concatenating to final array
            # print("fun-ELSE")
            finalsample = np.concatenate((finalsample, sample), axis=0)
            finalhotkey = np.concatenate((finalhotkey, hotkey), axis=0)

        # print(finalhotkey[-1, -1])
        # print(np.shape(sample), " ", np.shape(finalsample))
        # break  # restrict after one occurrence

    return np.shape(finalsample)[0], finalsample, finalhotkey


def sampling61(matfilePath : str, label : list, channel : list):

    matDict = loadmat(matfilePath)
    keys = list(matDict.keys())

    # for key in keys[3:]:
        # print(key)
    finalsample = []
    finalhotkey = []
    
    for vid in range(3,27):
        eegData = matDict[keys[vid]]

        df = DataFrame(eegData, index = channel)
        df = df.drop(index = channel[0])

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
 

def dirsample(path : list, label : list):
    # here we re loading all the 15 files of one directory
    # if we give the path of 1 then it will load the 15 files of it
    filename = listdir(path)
    sampleno, sample, hotkey = 0, np.zeros(0), np.zeros(0)

    for i in filename:
        total, rsample, rhotkey = sampling(path + i, label)
        # print(total)
        sampleno += total
        # print(np.shape(sample1))

        if not np.size(sample):
            # print("main()-IF")
            sample = rsample
            hotkey = rhotkey
        else:
            # print("main()-ELSE")
            sample = np.concatenate((sample, rsample), axis=0)
            hotkey = np.concatenate((hotkey, rhotkey), axis=0)
            # break
        print(i, sampleno, total)

    return sample, hotkey

def dirsample61(path : list, label : list, channel : list):
    filenames = listdir(path)
    sample, hotkey = np.zeros(0), np.zeros(0)

    for i in filenames:
        print(i)
        sample61, hotkey61 = sampling61(path + i, label, channel)

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