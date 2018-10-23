# Eleftheria Briakou
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import logging
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import time

class Autoencoder(nn.Module):
    '''
    Simple fully connected autoencoder with multiple layers
    '''
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(

            nn.Linear(33,100),
            nn.ReLU(True),
            nn.BatchNorm1d(100),

            nn.Linear(100, 40),
            nn.ReLU(True),
            nn.BatchNorm1d(40),

            nn.Linear(40, 15),
            nn.ReLU(True),
            nn.BatchNorm1d(15)

        )
        self.decoder = nn.Sequential(

            nn.Linear(15, 40),
            nn.ReLU(True),
            nn.BatchNorm1d(40),

            nn.Linear(40, 100),
            nn.ReLU(True),
            nn.BatchNorm1d(100),

            nn.Linear(100, 33),
            nn.ReLU(True),
            nn.BatchNorm1d(33)
        )

    def forward(self, x):
        '''
        Model forward pass

        :param input_: input examples
        :return: input reconstruction
        '''
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def evaluate(data_loader, model, device):
    """
    Evaluate the current model, get the accuracy for test set

    :param: data_loader: pytorch build-in data loader output
    :param: model: model to be evaluated
    :param: device: cpu of gpu
    :return: mae loss on test data
    """
    model.eval()

    loss_ = []
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
           data = batch.to(device)
           outputs = model.forward(data)
           loss_.append(F.l1_loss(outputs, data).data.numpy())

    return np.mean(loss_)

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

    for idx, batch in enumerate(train_data_loader):
        if len(batch)==1:
            continue
        data = batch.to(device)

        optimizer.zero_grad()
        out = model.forward(data)
        loss = criterion(out, data)
        loss.backward()
        optimizer.step()

    train_loss = evaluate(train_data_loader, model, device)
    end = time.time()

    return train_loss, (end - start)

def main():
    parser = argparse.ArgumentParser(description='Autoencoder (evaluare on three meter dataset)')
    parser.add_argument('--no-cuda', action='store_true', default=True)
    parser.add_argument('--data', type=str, default='./Three Meter/data.csv')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--visualize', type=bool, default=True)
    parser.add_argument('--analysis', type=bool, default=False)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    # Load/ Normalize data
    data = pd.read_csv(args.data)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    # Split data into training and test sets, seed the random process
    train, test = train_test_split(data.astype('float32'), test_size = 0.15, random_state = 200)

    model = Autoencoder()
    model.to(device)
    logging.info('Model architecture  -> %s' %(model))

    # Set optimizer and loss criterion of neural network
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.L1Loss()

    # Load batchified dataset
    # Choose random sampler for training set (serves as shuffling), test data are sampled in a sequence
    train_sampler = torch.utils.data.sampler.RandomSampler(train)
    test_sampler = torch.utils.data.sampler.SequentialSampler(test)
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, sampler=test_sampler)

    # Initialize list for keeping track of mae loss of training/test data over epochs
    training_mae, test_mae = [], []
    for epoch in range(args.num_epochs):

        # Random sample data in each epoch, and train
        train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, sampler=train_sampler)

        # Compute current loss/accuracy on test set
        curr_mae_train, time_ = training(model, train_loader, device, optimizer, criterion) #/train_loader.batch_size
        curr_mae_test = evaluate(test_loader, model, device) #/test_loader.batch_size
        test_mae.append(curr_mae_test)
        training_mae.append(curr_mae_train)

        logging.info('> Epoch: %d, training mae: %.5f, test mae: %.5f, time: %.5f'
                     % (epoch, curr_mae_train, curr_mae_test, time_))

    # Plot accuracy and loss over epochs
    if args.visualize:

        plt.plot(training_mae, color='blue', linewidth=3, alpha=0.3, marker='o', label = 'Training MAE')
        plt.plot(test_mae, color='red', linewidth=3, alpha=0.3, marker='o', label = 'Test MAE')
        plt.ylabel('MAE')
        plt.xlabel('epoch')
        plt.legend(loc='best')
        plt.xlim(0)
        plt.grid()
        plt.show()

    # Analysis of the trained network on three meter dataset
    if args.analysis:

        # Print 10 most poorly reconstructed example vectors
        mae = []
        test_loader = torch.utils.data.DataLoader(test, batch_size=1, sampler=test_sampler)
        for idx, batch in enumerate(test_loader):
            data = batch.to(device)

            outputs = model.forward(data)
            mae.append(np.mean(np.abs(outputs.detach().numpy()-data.detach().numpy())))
            poor = sorted(range(len(mae)), key=lambda i: mae[i])[-10:]

        for idx, ex in enumerate(poor):
            logging.info('Most poorly fitted example %d -> \n %s \n' % (idx, test_loader.dataset[ex]))


if __name__ == '__main__':
    main()