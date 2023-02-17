#Import necessary modules..
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import save_model, load_model

def plot_accuracy(results):
    plt.plot(results['accuracy'], label='accuracy')
    plt.plot(results['val_accuracy'], label='val_accuracy')

    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    plt.grid()
    plt.legend()
    plt.show()

def plot_loss(results):
    plt.plot(results['loss'], label='loss')
    plt.plot(results['val_loss'], label='val_loss')

    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.grid()
    plt.legend()
    plt.show()

def plot_all(results):
    plot_accuracy(results)
    plot_loss(results)

def save_model(model, path):
    save_model(model, path)

def load_model(filepath):
    model = load_model(filepath)
    return model
