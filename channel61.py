#functions imports..
from functions.sampling import sampling61, dirsample
from functions.pathlabelchannel import pathlabelchannel
from functions.cnn import runcnn2d

#package imports..
import numpy as np
import pandas as pd


path, label, channel = pathlabelchannel()


sample, hotkey = dirsample(path[0], label[0])

print(np.shape(sample), np.shape(hotkey))

model = runcnn2d(sample, hotkey, 0.8, 64, 150)
modelname = 'cnn2d_64_150_1'