#package imports..
import matplotlib.pyplot as plt

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