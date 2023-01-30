#package imports..
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def cnn2d(ncols : int, nrows : int, nplayers : int):
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


def runcnn2d61(sample : np.ndarray, hotkey : np.ndarray, ratio : float = 0.75, batch : int = 32, epoch : int = 80, channel : int = 61):

    X_train, X_test, y_train, y_test = train_test_split(sample, hotkey, train_size=ratio)

    model = cnn2d(channel, 1000, 1)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callback = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    results = model.fit(X_train, y_train, batch_size=batch, epochs=epoch, shuffle=True, validation_data=(X_test, y_test), callbacks=callback)

    modelplot_acc(results.history)
    modelplot_loss(results.history)
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
