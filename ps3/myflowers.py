# Eleftheria Briakou
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import logging
import time

class MyDataset(Dataset):
    """
    Pytorch data class for importing/tranforming the data
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        # Apply transformations on images
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)

def load_data(data_, labels_):
    '''
    Data loader

    :param data_: imput images
    :param labels_: labels of images
    :return: train and test data transformed
    '''
    # Import data, scale them to values between [0,1] and cast them to the appropriate types
    data = np.load(data_)
    labels = np.load(labels_)

    # Normalize each channel
    img_mean = np.mean(np.swapaxes(data / 255.0, 0, 1).reshape(3, -1), 1)
    img_std = np.std(np.swapaxes(data / 255.0, 0, 1).reshape(3, -1), 1)

    normalize = transforms.Normalize(mean=list(img_mean),std=list(img_std))

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.15, random_state=42)

    train = MyDataset(x_train, y_train, transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize]))

    test = MyDataset(x_test, y_test, transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        normalize,
    ]))

    return train, test

class Conv_Layer(nn.Module):
    '''
    Convolutional layer used as a unit the general neural network architecture
    '''
    def __init__(self, input_, output):
        super(Conv_Layer, self).__init__()

        self.conv = nn.Conv2d(in_channels=input_, out_channels=output, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=output)
        self.relu = nn.ReLU(True)

        self.net = nn.Sequential(self.conv, self.bn, self.relu)

    def forward(self, input_):
        '''
         Unit forward pass

        :param input_: input examples
        :return: convolutional layer outputs
        '''
        output = self.net(input_)
        return output

class Neural_Model(nn.Module):
    '''
    Convolutional neural network
    '''

    def __init__(self, shape, classes, dropout):
        super(Neural_Model, self).__init__()


        self.conv_32a = Conv_Layer(shape, 32)
        self.conv_32b = Conv_Layer(32, 32)

        self.conv_64a = Conv_Layer(32, 64)
        self.conv_64b = Conv_Layer(64, 64)

        self.conv_128a = Conv_Layer(64, 128)
        self.conv_128b = Conv_Layer(128,128)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(dropout)

        self.avgpool = nn.AvgPool2d(kernel_size=4)

        self.net = nn.Sequential(self.conv_32a, self.conv_32b, self.maxpool, self.drop,
                                 self.conv_64a, self.conv_64b, self.maxpool, self.drop,
                                 self.conv_128a, self.conv_128b, self.maxpool, self.drop, self.avgpool)

        self.fc = nn.Linear(in_features=128, out_features=classes)
        self.softmax_ = nn.Softmax()

    def forward(self, input_):
        '''
        Model forward pass

        :param input_: input examples
        :return: probability predictions
        '''
        output = self.net(input_)
        output = output.view(-1, output.shape[1]*output.shape[2]*output.shape[3])
        output = self.fc(output)
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
    test_acc = [] 
    loss = []
    for idx, batch in enumerate(data_loader):
        images = batch[0].to(device)
        labels = batch[1].to(device)

        # Predict classes using images from the test set
        outputs = model.forward(images)
        _, prediction = torch.max(outputs.data, 1)

        test_acc.append(torch.sum(prediction == labels.data).float())

        loss_ = criterion(outputs, labels)
        loss.append(loss_.data.cpu().numpy())

    return np.mean(test_acc)/data_loader.batch_size, np.mean(loss)

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
        data = batch[0].to(device)
        labels = batch[1].to(device)

        optimizer.zero_grad()
        out = model.forward(data)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

    accuracy, loss = evaluate(train_data_loader, model, device, criterion)
    end = time.time()

    return accuracy, loss, (end - start)

def main():
    parser = argparse.ArgumentParser(description='Convolutional neural network (evaluated on image recognition)')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--data', type=str, default='./Flowers/flower_imgs.npy')
    parser.add_argument('--labels', type=str, default='./Flowers/flower_labels.npy')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--analysis', type=bool, default=False)
    parser.add_argument('--visualize', type=bool, default=True)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    train, test = load_data(args.data, args.labels)

    shape_ = 3
    model = Neural_Model(shape_, args.num_classes, args.dropout)
    model.to(device)
    logging.info('Model architecture  -> %s' %(model))

    # Load batchified dataset
    # Choose random sampler for training set (serves as shuffling), test data are sampled in a sequence
    train_sampler = torch.utils.data.sampler.RandomSampler(train)
    test_sampler = torch.utils.data.sampler.SequentialSampler(test)
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, sampler=test_sampler)

    # Set optimizer and loss criterion of neural network
    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Initialize list for keeping track of loss/accuracy of training/test data over epochs
    tr_a, t_a, tr_l, t_l = [], [], [], []
    for epoch in range(args.num_epochs):

        # Random sample data in each epoch, and train
        train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, sampler=train_sampler)
        training_acc, training_loss, time_ = training(model, train_loader, device, optimizer, criterion)

        # Compute current loss/accuracy on test set
        test_acc, test_loss = evaluate(test_loader, model, device, criterion)

        logging.info(
            '> Epoch: %d, training accuracy: %.5f, test accuracy: %.5f, training loss: %.5f, test loss: %.5f, time: %.5f'
            % (epoch, training_acc, test_acc, training_loss, test_loss, time_))

        tr_a.append(training_acc)
        t_a.append(test_acc)
        tr_l.append(training_loss)
        t_l.append(test_loss)

    # Plot accuracy and loss over epochs
    if args.visualize:

        benchmark = len(tr_a)* [0.75]
        benchmark_extra = len(tr_a) * [0.8]
        plt.plot(tr_a, color='blue', linewidth=1, alpha=0.3, marker='o', label = 'training')
        plt.plot(t_a, color='red', linewidth=1, alpha=0.3, marker='o', label = 'test')
        plt.plot(benchmark, color='black', linewidth=2, alpha=0.7, dashes=[5, 2, 1, 3], label = 'Benchmark')
        plt.plot(benchmark_extra, color='green', linewidth=2, alpha=0.7, dashes=[1, 1, 1, 1], label = 'Benchmark Extra')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(loc='best')
        plt.xlim(0)
        plt.grid()
        plt.show()

        plt.clf()
        plt.plot(tr_l, color='blue', linewidth=1, alpha=0.3, marker='o', label='training')
        plt.plot(t_l, color='red', linewidth=1, alpha=0.3, marker='o', label='test')
        plt.ylabel('')
        plt.xlabel('epoch')
        plt.legend(loc='loss')
        plt.xlim(0)
        plt.grid()
        plt.show()

    # Analysis of the trained network on adult dataset
    if args.analysis:

        # Prin representative weights of output layer
        logging.info('Weights of output fully connected layer -> \n %s \n' %(model.fc.weight[0]))


        # Print 100 misclassified example vectors
        counter = 0
        test_loader = torch.utils.data.DataLoader(test, batch_size=1, sampler=test_sampler)
        for idx, batch in enumerate(test_loader):
            image = batch[0].to(device)
            label = batch[1].to(device)

            output = model.forward(image)
            _, prediction = torch.max(output.data, 1)

            if (prediction != label.data):
                # Prepare imafe for plotting
                image = batch[0].cpu().numpy()
                image = np.reshape(image, (3, 32, 32))

                # Plot
                plt.clf()
                plt.imshow((np.transpose(image, (1, 2, 0))))
                plt.show()
                counter += 1

            if counter == 100:
                break

if __name__ == '__main__':
    main()
