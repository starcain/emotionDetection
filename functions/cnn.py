# package imports
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split

# function imports
if __name__ == "__main__":
    from model import plot_all
else:
    from functions.model import plot_all


def create_cnn(ncols: int, nrows: int, nchannels: int) -> Sequential:
    model = Sequential()
    model.add(
        Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(ncols, nrows, nchannels)))
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


def train_cnn(sample: np.ndarray, hotkey: np.ndarray, nchannels: int = 62, train_ratio: float = 0.75, batch_size: int = 32, epochs: int = 80):
    X_train, X_test, y_train, y_test = train_test_split(sample, hotkey, train_size=train_ratio)

    model = create_cnn(nchannels, 1000, 1)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    # early_stop_callback = None

    results = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
                        validation_data=(X_test, y_test), callbacks=early_stop_callback)

    plot_all(results.history)
    return results
