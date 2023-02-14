# package imports
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import L2

# function imports
if __name__ == "__main__":
    from model import plot_all
else:
    from functions.model import plot_all

def create_cnn_reduce(ncols: int, nrows: int, nplayers: int) -> tf.keras.Model:
    model = Sequential([
        Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(ncols, nrows, nplayers)),
        MaxPooling2D(pool_size=(1, 5)),
        BatchNormalization(),
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(1, 3)),
        BatchNormalization(),
        Conv2D(filters=48, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(1, 3)),
        BatchNormalization(),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(1, 3)),
        BatchNormalization(),
        Conv2D(filters=80, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(1, 3)),
        BatchNormalization(),
        Flatten(),
        Dense(64, activation='relu', kernel_regularizer=L2(0.001)),
        Dropout(0.2),
        Dense(4, activation='softmax')
    ])

    print(model.summary())
    return model


def train_cnn_reduce(sample: np.ndarray, hotkey: np.ndarray, channel: int = 61, ratio: float = 0.75, batch_size: int = 32, epochs: int = 80):
    X_train, X_test, y_train, y_test = train_test_split(sample, hotkey, train_size=ratio)

    model = create_cnn_reduce(channel, 1000, 1)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

    results = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(X_test, y_test), callbacks=[callback])

    plot_all(results.history)
    return results
