from os import listdir
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.models import Sequential
from pickle import dump, load


def sampling(matfile, hlabel):
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


def plottocheck(data, x):  # function to check EEG signals using plotting
    # takes one sample and size of sample and shows plotting
    xaxis = list(range(x))
    plt.plot(xaxis, data)
    plt.show()


def modelplot_acc(results):
    plt.plot(results['accuracy'], label='accuracy')
    plt.plot(results['val_accuracy'], label='val_acc')

    plt.title('model_acccuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')

    plt.grid()
    plt.legend()
    plt.show()


def modelplot_loss(results):
    plt.plot(results['loss'], label='loss')
    plt.plot(results['val_loss'], label='val_loss')

    plt.title('model_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.grid()
    plt.legend()
    plt.show()


def npy_write(sample, hotkey, sname, hname):
    data = np.asarray(sample)
    np.save(sname + '.npy', data)

    data2 = np.load(sname + '.npy')
    print(np.shape(data2))

    data = np.asarray(hotkey)
    np.save(hname + '.npy', data)

    data2 = np.load(hname + '.npy')
    print(np.shape(data2))


def npy_read(samplenpy, hotkeynpy):
    # print(".npy file read DONE....")
    return np.load(samplenpy), np.load(hotkeynpy)


def dirsample(path, label):
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


def sample_shuffle(sno, sample, hotkey):
    # print("Shuffling started...")
    rarr = np.array(range(sno))
    np.random.shuffle(rarr)
    finalsample, finalhotkey = np.zeros((sno, 62, 1000)), np.zeros((sno, 4))

    for i in range(sno):
        pos = rarr[i]
        finalsample[i] = sample[pos]
        finalhotkey[i] = hotkey[pos]

    # print("Shuffling DONE...")
    return finalsample, finalhotkey


def cnn2d(ncols, nrows, nplayers):
    model = Sequential()
    model.add(
        Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(ncols, nrows, nplayers)))
    model.add(MaxPooling2D(pool_size=(2, 5)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 3)))
    model.add(Conv2D(filters=48, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 3)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 3)))
    model.add(Conv2D(filters=80, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 3)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='relu'))

    print(model.summary())
    return model


def runcnn2d(sample, hotkey, ratio, batch, epoch):
    sno = np.shape(sample)[0]
    finalsample, finalhotkey = sample_shuffle(sno, sample, hotkey)

    traintest = int(sno * ratio)
    x1 = finalsample[0:traintest, :, :]
    y1 = finalhotkey[0:traintest, :]
    x2 = finalsample[traintest:, :, :]
    y2 = finalhotkey[traintest:, :]

    model = cnn2d(62, 1000, 1)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    results = model.fit(x1, y1, batch_size=batch, epochs=epoch, validation_data=(x2, y2))

    modelplot_acc(results.history)
    # modelplot_loss(results.history)
    return results


def savemodel(model, modelname):
    dump(model, open(modelname + '.pkl', "wb"))


def readmodel(modelname):
    return load(open(modelname + '.pkl', "rb"))


'''
(Label) 0: neutral, 1: sad, 2: fear, 3: happy
path1 = "D:/project/eeg_raw_data/eeg_raw_data/1/"
label1 = [-1, -1, -1, 1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]
# 123020010121112322330303
path2 = "D:/project/eeg_raw_data/eeg_raw_data/2/"
label2 = [-1, -1, -1, 2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1]
# 213002023323201121030131
path3 = "D:/project/eeg_raw_data/eeg_raw_data/3/"
label3 = [-1, -1, -1, 1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
# 122133311210233023002010

sample1, hotkey1 = pj.dirsample(path1, label1)
sample2, hotkey2 = pj.dirsample(path2, label2)
sample3, hotkey3 = pj.dirsample(path3, label3)
'''

path = ["D:/project/eeg_raw_data/eeg_raw_data/1/",
        "D:/project/eeg_raw_data/eeg_raw_data/2/",
        "D:/project/eeg_raw_data/eeg_raw_data/3/"
        ]

label = [[-1, -1, -1, 1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
         [-1, -1, -1, 2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
         [-1, -1, -1, 1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
         ]

channel = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6',
           'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5',
           'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2',
           'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7',
           'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']

sample, hotkey = dirsample(path[0], label[0])

print(np.shape(sample), np.shape(hotkey))

model = runcnn2d(sample, hotkey, 0.8, 64, 150)
modelname = 'cnn2d_64_150_1'
savemodel(model, modelname)
