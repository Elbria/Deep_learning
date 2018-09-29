from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

def data_preparation(num_pixels):
    '''
    Data preparation function for the MNIST dataset.

    :param num_pixels: number of pixels of each image input
    :return: train and test features and labels after preprocessing
    '''

    x_train = np.load('MNIST/trainImages.npy')
    x_test = np.load('MNIST/testImages.npy')
    y_train = np.load('MNIST/trainLabels.npy')
    y_test = np.load('MNIST/testLabels.npy')

    # Use this initialization if you want to implement a perceptron
    #x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
    #x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

    x_train = x_train / 255
    x_test = x_test / 255
    return x_train, x_test, y_train, y_test

def outputs(history):
    '''
    Output plots of trained neural network.

    :param history: history of neural network
    '''

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    plt.show()

def model(num_pixels, num_classes, batch_size, epochs,  x_train, x_test, y_train, y_test):
    '''
    Main neural network model

    :param num_pixels: number of pixels for each image
    :param num_classes: number of output classes
    :param batch_size: batch size for the stochastic gradient descent algorithm
    :param epochs: number of epochs (passes over the entire training set)
    :param x_train: training set
    :param x_test: tesing set
    :param y_train: training set labels
    :param y_test: test set labels
    '''
    '''
    # These lines implement a simple perceptron
    model = Sequential()
    model.add(Dense(32, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs,
                        batch_size=batch_size, verbose=2)
    '''

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))
    outputs(history)

def main():
    '''
    Parameters definition for neural network
    '''

    # Hyperparameters definition
    batch_size = 128
    epochs = 1
    num_pixels = 28 * 28
    num_classes = 10

    x_train, x_test, y_train, y_test = data_preparation(num_pixels)
    model(num_pixels, num_classes, batch_size, epochs, x_train, x_test, y_train, y_test)

if __name__ == '__main__':
    main()



