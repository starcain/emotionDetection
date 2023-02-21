from scipy.io import loadmat
from os import listdir
from os.path import join
from numpy import shape
import matplotlib.pyplot as plt
from math import log10

from functions.pathlabelchannel import path

def get_data(path, path_index : int = 0, file_no : int = 0, vid_no: int = 3, channel_no : int = 0):
    files = listdir(path[path_index])
    matdict = loadmat(join(path[path_index] + files[file_no]))
    keys = list(matdict.keys())
    matdata = matdict[keys[vid_no]]
    print(shape(matdata))
    channeldata = matdata[channel_no]

    return channeldata

def plot(data : list):
    plt.plot(data)
    plt.grid()
    # plt.legend()
    plt.show()

channeldata = get_data(path)
length = len(channeldata)
print(length)
print(channeldata)
plot(channeldata)



