from keras.models import Sequential
import keras
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def data_preparation():
    '''
    Data preparation function for the Breast cancer dataset.

    :return: train and test features and labels after preprocessing
    '''

    x = pd.read_csv('Breast Cancer/breastCancerData.csv')
    y = pd.read_csv('Breast Cancer/breastCancerLabels.csv')

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train).astype('float32')
    x_test = sc.transform(x_test).astype('float32')

    y_train = keras.utils.to_categorical(y_train._values)  # these preserve dtype
    y_test = keras.utils.to_categorical(y_test._values)  # these preserve dtype

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

def model(num_features, num_classes, batch_size, epochs,  x_train, x_test, y_train, y_test):
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

    # These lines implement a simple perceptron
    model = Sequential()
    model.add(Dense(32, input_dim=num_features, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs,
                        batch_size=batch_size, verbose=2)

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))
    outputs(history)

def main():
    '''
    Parameters definition for neural network
    '''

    # Hyperparameters definition
    batch_size = 128
    epochs = 100
    num_features = 9
    num_classes = 2

    x_train, x_test, y_train, y_test = data_preparation()
    model(num_features, num_classes, batch_size, epochs, x_train, x_test, y_train, y_test)

if __name__ == '__main__':
    main()



