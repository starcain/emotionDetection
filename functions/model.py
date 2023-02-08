#package imports..
import matplotlib.pyplot as plt
from keras.models  import load_model

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

def plotAll(results):
    modelplot_acc(results)
    modelplot_loss(results)

def savemodel(model, path):
    model.save(path)

def loadModel(filepath):
    model = load_model(filepath)
    return model