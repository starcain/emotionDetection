#functions imports..
from functions.sampling import dirsample61
from functions.pathlabelchannel import pathlabelchannel
from functions.cnn import runcnn2d61

#package imports..
import numpy as np

path, label, channel = pathlabelchannel()


sample, hotkey = dirsample61(path[0], label[0], channel)

model = runcnn2d61(sample, hotkey, 0.8, batch=32, epoch=150, channel=61)

