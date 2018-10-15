import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import logging
import time

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

class Conv_Layer(nn.Module):
    def __init__(self, input, output, dropout_):
        super(Conv_Layer, self).__init__()

        self.conv = nn.Conv2d(in_channels=input, out_channels=output, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=output)
        self.relu = nn.ReLU()
        #self.drop = torch.nn.Dropout(dropout_)

        self.net = nn.Sequential(self.conv, self.bn, self.relu)
        #self.net = nn.Sequential(self.conv, self.relu, self.drop)

    def forward(self, input):
        output = self.net(input)
        return output

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])

class Neural_Model(nn.Module):
    def __init__(self, shape, classes, nn_dropout):
        super(Neural_Model, self).__init__()

        self.dropout = nn_dropout

        self.conv1 = Conv_Layer(shape, 32, self.dropout)
        self.conv2 = Conv_Layer(32, 32, self.dropout)
        self.conv3 = Conv_Layer(32, 32, self.dropout)


        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.conv4 = Conv_Layer(32, 64, self.dropout)
        self.conv5 = Conv_Layer(64, 64, self.dropout)
        self.conv6 = Conv_Layer(64, 64, self.dropout)
        self.conv7 = Conv_Layer(64, 64, self.dropout)

        #self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.conv8 = Conv_Layer(64, 128, self.dropout)
        self.conv9 = Conv_Layer(128, 128, self.dropout)
        self.conv10 = Conv_Layer(128, 128, self.dropout)
        self.conv11 = Conv_Layer(128, 128, self.dropout)


        #self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.conv12 = Conv_Layer(128, 128, self.dropout)
        self.conv13 = Conv_Layer(128, 128, self.dropout)
        self.conv14 = Conv_Layer(128, 128, self.dropout)


        self.avgpool = nn.AvgPool2d(kernel_size=4)

        self.net = nn.Sequential(self.conv1, self.conv2, self.maxpool)

                                 #self.conv8, self.conv9, self.conv10, self.conv11, self.maxpool,
                                 #self.conv12, self.conv13, self.conv14, self.avgpool)

        self.fc = nn.Linear(in_features=32*16*16, out_features=classes)
        self.softmax_ = nn.Softmax()

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, output.shape[1]*output.shape[2]*output.shape[3])
        output = self.fc(output)
        return output

def evaluate(data_loader, model, device):
    """
    evaluate the current model, get the accuracy for dev/test set
    Keyword arguments:
    data_loader: pytorch build-in data loader output
    model: model to be evaluated
    device: cpu of gpu
    """
    model.eval()
    error = 0
    num_examples = 0
    for idx, batch in enumerate(data_loader):
        images = batch[0].to(device)
        labels = batch[1]

        logits = model.forward(images)
        top_n, top_i = logits.topk(1)
        num_examples += images.size(0)
        error += torch.nonzero(top_i.squeeze() - torch.LongTensor(labels)).size(0)
    accuracy = 1 - error / num_examples
    print('accuracy', accuracy)
    return accuracy

def train_(args, model, train_data_loader, dev_data_loader, accuracy, device):

    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    print_loss_total = 0
    epoch_loss_total = 0
    start = time.time()

    for idx, batch in enumerate(train_data_loader):
        images = batch[0].to(device)
        labels = batch[1]

        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        print_loss_total += loss.data.numpy()
        epoch_loss_total += loss.data.numpy()

        if idx % args.checkpoint == 0 and idx > 0:
            print_loss_avg = print_loss_total / args.checkpoint

            print('number of steps: %d, loss: %.5f time: %.5f' % (idx, print_loss_avg, time.time() - start))
            print_loss_total = 0
            curr_accuracy = evaluate(dev_data_loader, model, device)
            if accuracy < curr_accuracy:
                torch.save(model, args.save_model)
                accuracy = curr_accuracy
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Question Type')
    parser.add_argument('--no-cuda', action='store_true', default=True)
    parser.add_argument('--data', type=str, default='./Flowers/flower_imgs.npy')
    parser.add_argument('--labels', type=str, default='./Flowers/flower_labels.npy')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--checkpoint', type=int, default=10)
    parser.add_argument('--save-model', type=str, default='q_type.pt')
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    data = np.load(args.data)
    data = data.reshape(data.shape[0], 3, 32, 32).astype('float32')
    data /= 256
    labels = np.load(args.labels)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.15, random_state=42)
    #x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.15, random_state=42)
    x_dev, y_dev = x_test, y_test
    train = Dataset_(x_train, y_train)
    test = Dataset_(x_test, y_test)
    dev = Dataset_(x_dev, y_dev)

    logging.info('Splitting data into training (%d) test (%d) and development (%d) sets.'
                 %(len(y_train), len(y_test), len(y_dev)))

    shape_ = 3
    model = Neural_Model(shape_, args.num_classes, 0.25)
    model.to(device)
    logging.info('Model architecture  -> %s' %(model))

    # Load batchified dataset
    train_sampler = torch.utils.data.sampler.RandomSampler(train)
    dev_sampler = torch.utils.data.sampler.SequentialSampler(dev)
    dev_loader = torch.utils.data.DataLoader(dev, batch_size=args.batch_size, sampler=dev_sampler)

    accuracy = 0
    for epoch in range(args.num_epochs):
        print('start epoch %d' % epoch)
        train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, sampler=train_sampler)
        accuracy = train_(args, model, train_loader, dev_loader, accuracy, device)


    print('start testing:\n')

    test_sampler = torch.utils.data.sampler.SequentialSampler(test)
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, sampler=test_sampler)
    evaluate(test_loader, model, device)


if __name__ == '__main__':
    main()