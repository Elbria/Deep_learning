from keras.models import Sequential
import numpy as np
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def data_preparation():
    '''
    Data preparation function for the MNIST dataset.

    :return: train and test features and labels after preprocessing
    '''

    x_train = np.load('MNIST/trainImages.npy')
    x_test = np.load('MNIST/testImages.npy')
    y_train = np.load('MNIST/trainLabels.npy')
    y_test = np.load('MNIST/testLabels.npy')

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

    x_train = x_train / 255
    x_test = x_test / 255

    return x_train, x_test, y_train, y_test

def outputs(hist_):
    '''
    Output plots of trained neural network.

    :param history: history of neural network
    '''
    colors = ['b', 'm', 'y', 'b', 'r', 'g','k']

    # Plot accuracy
    for i, history in enumerate(hist_):
        plt.plot(history.history['acc'], linestyle='dashed', color=colors[i], label = 'Training')
        plt.plot(history.history['val_acc'], color=colors[i], linewidth=3, label = 'Test')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.ylim(0.95,1)
    plt.xlim(0)
    plt.grid()
    plt.show()

    plt.clf()
    # Plot loss
    for i, history in enumerate(hist_):
        plt.plot(history.history['loss'], linestyle='dashed', color=colors[i], label = 'Training')
        plt.plot(history.history['val_loss'], color=colors[i], linewidth=3,  label='Test')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.ylim(0,0.1)
    plt.xlim(0)
    plt.grid()
    plt.show()


def simple(i, j, k, n, batch_size, epochs, num_classes, x_train, x_test, y_train, y_test):
    model = Sequential()
    model.add(Conv2D(n, (i, i), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(j, j)))
    model.add(Dropout(k))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(k))
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
    return (100 - scores[1] * 100), history

def baseline(num_classes, batch_size, epochs,  x_train, x_test, y_train, y_test, parameter):
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

    if parameter=='filter':
        history = []
        for i in range(3,8,2):
            t, h = simple(i, 2, 0.2, 32, batch_size, epochs, num_classes, x_train, x_test, y_train, y_test)
            history.append(h)

    if parameter=='pooling':
        history = []
        for j in range(2,5):
            t, h = simple(7, j, 0.2, 32, batch_size, epochs, num_classes, x_train, x_test, y_train, y_test)
            history.append(h)

    if parameter=='dropout':
        history = []
        for k in range(0,5):
            t, h = simple(7, 2, k*0.1, 32, batch_size, epochs, num_classes, x_train, x_test, y_train, y_test)
            history.append(h)

    if parameter=='num_filter':
        history = []
        num = [16, 32, 64]
        for n in num:
            t, h = simple(7, 2, 0.2, n, batch_size, epochs, num_classes, x_train, x_test, y_train, y_test)
            history.append(h)

    outputs(history)

def model(num_classes, batch_size, epochs,  x_train, x_test, y_train, y_test, param):
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

    model = Sequential()
    model.add(Conv2D(32, (7, 7), input_shape=(28, 28, 1), activation='relu'))
    if param=='second':
        model.add(Conv2D(64, (7, 7), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
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
    outputs([history])

    predictions = np.argmax(model.predict(x_test), axis=1)
    ground_truth = np.nonzero(y_test)[1]
    incorrects_idx = [ground_truth[i] for i in range(len(ground_truth)) if predictions[i]!=ground_truth[i]]
    incorrects = np.nonzero(predictions - ground_truth)[0]

    # Plot 20 misclassified images
    i=0
    for sample in incorrects:
        if i<20:
            two_d = (np.reshape(x_test[sample], (28, 28)) * 255).astype(np.uint8)
            plt.imshow(two_d, cmap=plt.get_cmap('gray'))
            plt.title('Ground truth: ' + str(ground_truth[sample]) + ', Prediction: ' + str(predictions[sample]))
            plt.show()
            i+=1

    # Plot distribution of mosclasiffied examples over classes
    plt.clf()
    sns.distplot(incorrects_idx)
    plt.grid()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='CNN experimentation for MNIST dataset')
    parser.add_argument('option', help='choose configuration', choices=['complex', 'baseline'])
    parser.add_argument('parameter', help='choose parameter for exploration',
                        choices=['filter','pooling','dropout','num_filter', 'second', 'predefined'], type=str)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Hyperparameters definition
    batch_size = 128
    epochs = 25
    num_classes = 10

    x_train, x_test, y_train, y_test = data_preparation()

    if args.option=='complex':
        model(num_classes, batch_size, epochs, x_train, x_test, y_train, y_test, args.parameter)
    elif args.option=='baseline':
        baseline(num_classes, batch_size, epochs, x_train, x_test, y_train, y_test, args.parameter)

if __name__ == '__main__':
    main()



