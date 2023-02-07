#package imports..
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#function imports..
from functions.modelplot import modelplot_acc, modelplot_loss

def cnn2d1x(ncols : int, nrows : int, nplayers : int):
    model = Sequential()
    model.add(
        Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(ncols, nrows, nplayers)))
    model.add(MaxPooling2D(pool_size=(1, 5)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(1, 3)))
    model.add(Conv2D(filters=48, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(1, 3)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(1, 3)))
    model.add(Conv2D(filters=80, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(1, 3)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    print(model.summary())
    return model

def runcnn2d61(sample : np.ndarray, hotkey : np.ndarray, channel : int = 61, ratio : float = 0.75, batch : int = 32, epoch : int = 80):

    X_train, X_test, y_train, y_test = train_test_split(sample, hotkey, train_size=ratio)

    model = cnn2d1x(channel, 1000, 1)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    # callback = None

    results = model.fit(X_train, y_train, batch_size=batch, epochs=epoch, shuffle=True, validation_data=(X_test, y_test), callbacks=callback)

    modelplot_acc(results.history)
    modelplot_loss(results.history)
    return results