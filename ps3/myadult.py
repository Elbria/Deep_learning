# Eleftheria Briakou
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import logging
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

class Mydataset(Dataset):
    """
    Pytorch data class for importing the data
    """
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

class Neural_Network(nn.Module):
    '''
    Fully connected neural network with one hidden layer
    '''
    def __init__(self, hidden):
        super(Neural_Network, self).__init__()

        out_ = 15
        if hidden:
            out_ = 30
        self.input_layer = nn.Linear(in_features=67, out_features=out_)
        self.hidden_layer = nn.Linear(in_features=30, out_features=15)
        self.output_layer = nn.Linear(in_features=15, out_features=1)
        self.out_act = nn.Sigmoid()

        if hidden:
            self.net = nn.Sequential(self.input_layer, self.hidden_layer, self.output_layer, self.out_act)
        else:
            self.net = nn.Sequential(self.input_layer, self.output_layer, self.out_act)

    def forward(self, input_):
        '''
        Model forward pass

        :param input_: input examples
        :return: probability predictions
        '''
        output = self.net(input_)
        return output

def evaluate(data_loader, model, device, criterion):
    """
    Evaluate the current model, get the accuracy for test set

    :param: data_loader: pytorch build-in data loader output
    :param: model: model to be evaluated
    :param: device: cpu of gpu
    :param criterion: criterion that computes loss
    :return: accuracy and loss on test data
    """

    model.eval()
    correct = 0
    loss = []
    for idx, batch in enumerate(data_loader):
        data = batch[0].to(device)
        labels = batch[1].to(device)

        outputs = model.forward(data).view(1,-1)
        # Threshold tensor into binary values based on
        # probability returned by the network
        predicted = torch.round(outputs).byte()
        correct += (predicted == labels.byte()).sum()

        loss_ = criterion(outputs, labels.float())
        loss.append(loss_.data.cpu().numpy())

    accuracy = correct.float() / data_loader.dataset.labels.size
    return accuracy, np.mean(loss)

def training(model, train_data_loader, device, optimizer, criterion):
    '''
    Main function that trains the model using optimizer according to criterion over the train_data_loader data
    After training is completed, the current performance on training data is computed

    :param model: model to be trained
    :param train_data_loader: pytorch build-in data loader output for training examples
    :param device: cpu of gpu
    :param optimizer: optimization algorithm
    :param criterion: criterion that computes loss
    :return: accuracy and loss on training data after all batches are parsed, time needed for completion
    '''

    model.train()
    start = time.time()

    # For each random batch in training data
    for idx, batch in enumerate(train_data_loader):
        # Move data to device
        datum = batch[0].to(device)
        labels = batch[1].to(device)

        optimizer.zero_grad()
        out = model.forward(datum)
        loss = criterion(out, labels.float())
        loss.backward()
        optimizer.step()

    # Compute accuracy and loss on training data
    accuracy, loss = evaluate(train_data_loader, model, device, criterion)
    end = time.time()

    return accuracy, loss, (end - start)

def main():
    parser = argparse.ArgumentParser(description='Fully Connected Neural Network (evaluated on Adult dataset)')
    parser.add_argument('hidden', type=bool)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--data', type=str, default='./Adult/data.npy')
    parser.add_argument('--labels', type=str, default='./Adult/labels.npy')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-epochs', type=int, default=25)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--analysis', type=bool, default=True)
    parser.add_argument('--visualize', type=bool, default=True)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    # Import data, scale them to values between [0,1] and cast them to the appropriate types
    data = np.load(args.data)
    data = data.astype('float32')

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    labels = np.load(args.labels)
    labels = labels.astype('int64')

    # Split data into training and test sets, seed the random process
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.15, random_state=10)
    logging.info('Split data into training and test sets of %d and %d examples, respectively.' % (len(y_train), len(y_test)))
    train = Mydataset(x_train, y_train)
    test = Mydataset(x_test, y_test)


    model = Neural_Network(args.hidden)
    model.to(device)
    logging.info('Model architecture  -> %s' %(model))

    # Load batchified dataset
    # Choose random sampler for training set (serves as shuffling), test data are sampled in a sequence
    train_sampler = torch.utils.data.sampler.RandomSampler(train)
    test_sampler = torch.utils.data.sampler.SequentialSampler(test)
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, sampler=test_sampler)

    # Set optimizer and loss criterion of neural network
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()

    # Initialize list for keeping track of loss/accuracy of training/test data over epochs
    tr_a, t_a, tr_l, t_l = [], [], [], []
    for epoch in range(args.num_epochs):

        # Random sample data in each epoch, and train
        train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, sampler=train_sampler)
        training_acc, training_loss, time_ = training(model, train_loader, device, optimizer, criterion)

        # Compute current loss/accuracy on test set
        test_acc, test_loss = evaluate(test_loader, model, device, criterion)

        logging.info('> Epoch: %d, training accuracy: %.5f, test accuracy: %.5f, training loss: %.5f,'
                     ' test loss: %.5f, time: %.5f'
                     % (epoch, training_acc, test_acc, training_loss, test_loss, time_))

        tr_a.append(training_acc)
        t_a.append(test_acc)
        tr_l.append(training_loss)
        t_l.append(test_loss)

    # Plot accuracy and loss over epochs
    if args.visualize:

        benchmark = len(tr_a)* [0.825]
        plt.plot(tr_a, color='blue', linewidth=2, alpha=0.5, marker='o', label='training')
        plt.plot(t_a, color='red', linewidth=2, alpha=0.5, marker='o', label='test')
        plt.plot(benchmark, color='black', linewidth=2, alpha=0.5, dashes=[5, 2, 1, 3], label='Benchmark')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(loc='best')
        plt.xlim(0)
        plt.grid()
        plt.show()

        plt.plot(tr_l, color='blue', linewidth=3, alpha=0.5, marker='o', label='training')
        plt.plot(t_l, color='red', linewidth=3, alpha=0.5, marker='o', label='test')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='loss')
        plt.xlim(0)
        plt.grid()
        plt.show()

    # Analysis of the trained network on adult dataset
    if args.analysis:

        # Print weights
        logging.info('Weights of input layer -> \n %s \n' %(model.input_layer.weight[0]))
        if args.hidden:
            logging.info('Weights of hidden layer -> \n %s \n' %(model.hidden_layer.weight[0]))
        logging.info('Weights of output layer -> \n %s \n' %(model.output_layer.weight[0]))

        # Print 10 misclassified example vectors
        counter = 0
        test_loader = torch.utils.data.DataLoader(test, batch_size=1, sampler=test_sampler)
        for idx, batch in enumerate(test_loader):
            data = batch[0].to(device)
            labels = batch[1].to(device)

            outputs = model.forward(data).view(1, -1)
            predicted = torch.round(outputs).byte()
            if (predicted != labels.byte()):
                counter += 1
                logging.info('Misclassified example %d -> \n %s \n' %(idx, data))
            if counter == 10:
                break

if __name__ == '__main__':
    main()
