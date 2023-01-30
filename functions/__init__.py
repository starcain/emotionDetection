from functions.pathlabelchannel import modelTestPLC, channelRemove
from functions.sampling import dirsample61
from functions.cnn import runcnn2d61

def modelTest(index : int, dropIndex : list, ratio : int, batch : int, epoch : int):
    path, label, channel = modelTestPLC(index)

    sample, hotkey = dirsample61(path, label, channel, dropIndex)

    model = runcnn2d61(sample, hotkey, ratio, batch, epoch, 62-len(dropIndex))

    return model 