#package imports..
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.models import Sequential
import numpy as np 
import matplotlib.pyplot as plt

#function._ imports..
from functions.sampling import sampleShuffle

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
    finalsample, finalhotkey = sampleShuffle(sno, sample, hotkey)

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
