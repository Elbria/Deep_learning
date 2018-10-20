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

class Dataset_(Dataset):
    """
    Pytorch data class for classification data
    """
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

class Neural_Model(nn.Module):
    def __init__(self):
        super(Neural_Model, self).__init__()

        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_features=67, out_features=30)
        self.fc2 = nn.Linear(in_features=30, out_features=15)
        self.fc3 = nn.Linear(in_features=15, out_features=1)
        self.relu = nn.ReLU()
        self.out_act = nn.Sigmoid()

        self.net = nn.Sequential(self.fc1, self.fc2, self.fc3, self.out_act)

    def forward(self, input):
        output = self.net(input)
        return output

def evaluate(data_loader, model, device, criterion):
    """
    evaluate the current model, get the accuracy for dev/test set
    Keyword arguments:
    data_loader: pytorch build-in data loader output
    model: model to be evaluated
    device: cpu of gpu
    """
    model.eval()
    correct = 0
    loss = []
    for idx, batch in enumerate(data_loader):
        data = batch[0].to(device)
        labels = batch[1].to(device)

        outputs = model.forward(data).view(1,-1)
        predicted = torch.round(outputs).byte()
        correct += (predicted == labels.byte()).sum()

        loss_ = criterion(outputs, labels.float())
        loss.append(loss_.data.numpy())

    accuracy = correct.float() / data_loader.dataset.labels.size
    return accuracy, np.mean(loss)

def train_(model, train_data_loader, device, optimizer, criterion):

    model.train()
    start = time.time()

    for idx, batch in enumerate(train_data_loader):
        datum = batch[0].to(device)
        labels = batch[1].to(device)

        optimizer.zero_grad()
        out = model.forward(datum)
        loss = criterion(out, labels.float())
        loss.backward()
        optimizer.step()

    accuracy, loss = evaluate(train_data_loader, model, device, criterion)
    end = time.time()

    return accuracy, loss, (end - start)

def main():
    parser = argparse.ArgumentParser(description='Question Type')
    parser.add_argument('--no-cuda', action='store_true', default=True)
    parser.add_argument('--data', type=str, default='./Adult/data.npy')
    parser.add_argument('--labels', type=str, default='./Adult/labels.npy')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=25)
    parser.add_argument('--visualize', type=bool, default=True)
    parser.add_argument('--save-model', type=str, default='q_type.pt')
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    data = np.load(args.data)
    data = data.astype('float32')

    # Create scaler
    scaler = MinMaxScaler()
    # Transform the feature
    data = scaler.fit_transform(data)

    labels = np.load(args.labels)
    labels = labels.astype('int64')

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.15, random_state=10)
    train = Dataset_(x_train, y_train)
    test = Dataset_(x_test, y_test)

    logging.info('Splitting data into training (%d) test (%d) sets.'
                 %(len(y_train), len(y_test)))

    model = Neural_Model()
    model.to(device)
    logging.info('Model architecture  -> %s' %(model))

    # Load batchified dataset
    train_sampler = torch.utils.data.sampler.RandomSampler(train)
    test_sampler = torch.utils.data.sampler.SequentialSampler(test)
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, sampler=test_sampler)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    criterion = nn.BCELoss()

    tr_a, t_a, tr_l, t_l = [], [], [], []
    for epoch in range(args.num_epochs):
        train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, sampler=train_sampler)
        training_acc, training_loss, time_ = train_(model, train_loader, device, optimizer, criterion)
        test_acc, test_loss = evaluate(test_loader, model, device, criterion)

        logging.info('> Epoch: %d, training accuracy: %.5f, test accuracy: %.5f, training loss: %.5f, test loss: %.5f, time: %.5f'
                     % (epoch, training_acc, test_acc, training_loss, test_loss, time_))

        tr_a.append(training_acc)
        t_a.append(test_acc)
        tr_l.append(training_loss)
        t_l.append(test_loss)

    if args.visualize:

        plt.plot(tr_a, color='blue', linewidth=3, alpha=0.3, marker='o', label = 'training')
        plt.plot(t_a, color='red', linewidth=3, alpha=0.3, marker='o', label = 'test')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(loc='best')
        plt.xlim(0)
        plt.grid()
        plt.show()

        plt.plot(tr_l, color='blue', linewidth=3, alpha=0.3, marker='o', label='training')
        plt.plot(t_l, color='red', linewidth=3, alpha=0.3, marker='o', label='test')
        plt.ylabel('')
        plt.xlabel('epoch')
        plt.legend(loc='loss')
        plt.xlim(0)
        plt.grid()
        plt.show()

if __name__ == '__main__':
    main()
