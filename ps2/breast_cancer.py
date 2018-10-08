from keras.models import Sequential
import keras
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import logging
import seaborn as sns
import numpy as np

def data_preparation():
    '''
    Data preparation function for the Breast cancer dataset.

    :return: train and test features and labels after preprocessing
    '''

    x = pd.read_csv('Breast Cancer/breastCancerData.csv')
    y = pd.read_csv('Breast Cancer/breastCancerLabels.csv')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

    # Distributions of data over candidate classes
    plt.clf()
    sns.distplot(y)
    plt.grid()
    plt.title('Distribution of dataset')
    plt.show()

    plt.clf()
    sns.distplot(y_train)
    plt.grid()
    plt.title('Distribution of training data')
    plt.show()

    plt.clf()
    sns.distplot(y_test)
    plt.grid()
    plt.title('Distribution of testing data')
    plt.show()

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train).astype('float32')
    x_test = sc.transform(x_test).astype('float32')

    y_train = keras.utils.to_categorical(y_train._values)
    y_test = keras.utils.to_categorical(y_test._values)

    return x_train, x_test, y_train, y_test

def outputs(hist_, a):
    '''
    Output plots of trained neural network.

    :param history: history of neural network
    :param a: auxiliary list for labeling plots
    '''
    colors = ['b', 'm', 'y', 'c', 'r', 'g','k']

    # Plot accuracy
    for i, history in enumerate(hist_):
        plt.plot(history.history['acc'], linestyle='dashed', color=colors[i], label=a[i])
        if a[i]=='Training':
            plt.plot(history.history['val_acc'], color=colors[i], linewidth=3, label='Test')
            plt.ylim(0.85, 1)
        else:
            plt.plot(history.history['val_acc'], color=colors[i], linewidth=3)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.xlim(0)
    plt.grid()
    plt.show()

    plt.clf()
    # Plot loss
    for i, history in enumerate(hist_):
        plt.plot(history.history['loss'], linestyle='dashed', color=colors[i], label=a[i])
        if a[i]=='Training':
            plt.plot(history.history['val_loss'], color=colors[i], linewidth=3, label='Test')
            plt.ylim(0, 0.5)
        else:
            plt.plot(history.history['val_loss'], color=colors[i], linewidth=3)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.xlim(0)
    plt.grid()
    plt.show()

def model(nodes, num_features, layers, num_classes, batch_size, epochs,  x_train, x_test, y_train, y_test):
    '''
    Main neural network model

    :param nodes: number of nodes in the hidden layer
    :param num_features: number of features
    :param: layers: number of layers
    :param num_classes: number of output classes
    :param batch_size: batch size for the stochastic gradient descent algorithm
    :param epochs: number of epochs (passes over the entire training set)
    :param x_train: training set
    :param x_test: tesing set
    :param y_train: training set labels
    :param y_test: test set labels
    '''

    model = Sequential()
    model.add(Dense(nodes, input_dim=num_features, kernel_initializer='normal', activation='relu'))
    # Extra hidden layers
    for i in range(layers-1):
        model.add(Dense(int(nodes/(i+2)), kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs,
                        batch_size=batch_size, verbose=2)

    scores = model.evaluate(x_test, y_test, verbose=0)

    # Representative outputs of weights and biases of the last layer
    final_wb = model.layers[-1].get_weights()
    logging.info('Print outs of weights (final layer): %s' % (final_wb[0]))
    logging.info('Print outs for biases (final layer): %s' % (final_wb[1]))

    logging.info("Baseline Error: %.2f%%" % (100-scores[1]*100))
    return history, model

def tune(option, nodes, num_features, layers, num_classes, batch_size, epochs, x_train, x_test, y_train, y_test):
    '''
    This function implements tuning of specific hyper-parameters

    :param option: hyperparameter to tune
    :param nodes: number of nodes
    :param num_features: number of features
    :param layers: number of layers
    :param num_classes: number of classes
    :param batch_size: batch size for training
    :param epochs: number of epochs
    :param x_train: training set
    :param x_test: tesing set
    :param y_train: training set labels
    :param y_test: test set labels
    '''

    results = []
    if option == 'batch_size':
        for batch in batch_size:
            history_, model_ = model(nodes, num_features, layers, num_classes, batch, epochs, x_train, x_test, y_train, y_test)
            results.append(history_)
        outputs(results, batch_size)

    if option == 'nodes':
        for node in nodes:
            history_, model_ = model(node, num_features, layers, num_classes, batch_size, epochs, x_train, x_test, y_train, y_test)
            results.append(history_)
        outputs(results, nodes)

    if option == 'layers':
        for layer in layers:
            history_, model_ = model(nodes, num_features, layer, num_classes, batch_size, epochs, x_train, x_test, y_train, y_test)
            results.append(history_)
        outputs(results, layers)

def main():
    '''
    Parameters definition for neural network
    '''
    parser = argparse.ArgumentParser(description='ANN experimentation for breast cancer dataset')
    parser.add_argument('option', help='choose configuration', choices=['final', 'tuning'])
    parser.add_argument('parameter', help='choose parameter for exploration',
                        choices=['batch_size', 'nodes', 'layers', 'predefined'], type=str)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


    # Hyperparameters definition (for final configuration)
    batch_size = 100
    epochs = 100
    num_features = 9
    num_classes = 2
    nodes = 30
    layers = 1

    x_train, x_test, y_train, y_test = data_preparation()
    # If this option is chosen, hyperparameter tuning is implemented
    if args.option=='tuning':
        # Tune batch size
        if args.parameter == 'batch_size':
            lenght_ = len(y_train)
            batch_size = [1, int(lenght_/10), int(lenght_/5), int(lenght_/2), len(y_train)]
            tune(args.parameter, nodes, num_features, layers, num_classes, batch_size,
                 epochs, x_train, x_test, y_train, y_test)

        # Tune of number nodes
        if args.parameter == 'nodes':
            nodes = [5, 10, 20, 30, 100]
            tune(args.parameter, nodes, num_features, layers, num_classes, batch_size,
            epochs, x_train, x_test, y_train, y_test)

        # Tune number of layers
        if args.parameter == 'layers':
            layers = [1, 2, 3]
            tune(args.parameter, nodes, num_features, layers, num_classes, batch_size,
                 epochs, x_train, x_test, y_train, y_test)

    elif args.option == 'final':
        history, model_ = model(nodes, num_features, layers, num_classes, batch_size,
                             epochs, x_train, x_test, y_train, y_test)
        outputs([history], ['Training'])

        predictions = np.argmax(model_.predict(x_test), axis=1)
        ground_truth = np.nonzero(y_test)[1]
        incorrects_idx = [ground_truth[i] for i in range(len(ground_truth)) if predictions[i] != ground_truth[i]]
        logging.info('Misclassified data: %s' %(incorrects_idx))

        # Plot distribution of misclassified examples over classes
        plt.clf()
        sns.distplot(incorrects_idx)
        plt.grid()
        plt.title('Distribution of misclassified data')
        plt.show()

if __name__ == '__main__':
    main()



