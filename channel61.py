#functions imports..
from functions.sampling import sampling61, dirsample61
from functions.pathlabelchannel import pathlabelchannel
from functions.cnn import runcnn2d

#package imports..
import numpy as np
import pandas as pd
from os import listdir

path, label, channel = pathlabelchannel()


sample, hotkey = dirsample61(path[0], label[0], channel)

