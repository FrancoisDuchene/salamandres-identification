from keras.datasets import mnist
from keras import models, layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def exemple_1():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    # architecture du reseau
    network = models.Sequential()
    network.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))
    # compilation du reseau
    network.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    # preparation des donnees d'images
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255
    # preparation des etiquettes
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    # on entraine le reseau
    network.fit(train_images, train_labels, epochs=5, batch_size=128)
    # on verifie que le modele fonctionne avec l'ensemble de test
    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print('test_acc:', test_acc)
    print('test_loss:', test_loss)


def exemple_2():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    digit = train_images[3]
    plt.imshow(digit, cmap=plt.cm.binary)

    plt.show()
    print(train_labels[8])

    my_slice = train_images[10:100, :, :]
    print(my_slice.shape)


if __name__ == '__main__':
    exemple_2()
