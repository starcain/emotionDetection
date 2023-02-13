i = 1
while i>0:
    dropIndex = [i]
    print(dropIndex)
    sample, hotkey = dirsample61(path[0], label[0], channel, dropIndex)
    model = runcnn2d61(sample, hotkey, 61)
    i-=1
    del dropIndex, sample, hotkey, model