# Eleftheria Briakou
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt

class Neural_Network:

    def __init__(self, layers, nodes, eta, iterations, active, problem):
        '''
        Initialize a neural network

        :param layers: number of layers
        :param nodes: number of nodes for each hidden layer
        :param eta: step size for updates
        :param iterations: number of iterations
        :param active: controls the activation function of hidden layers (0 for identity function, 1 for ReLu)
        :param problem: type of problem ('regression' or 'classification')
        '''
        logging.info('Initializing neural network')
        self.weights = {}
        self.biases = {}
        self.layers = layers
        self.eta = eta
        self.iterations = iterations
        self.active = active
        self.problem = problem

        #  Initialize weights that connect the layers and biases of each node at random values
        for l in range(self.layers):
            self.weights[l] = np.random.randn(nodes[l],nodes[l+1]).T
            self.biases[l] = np.random.randn(nodes[l+1], 1).T

    def feed_forward(self, input):
        '''
        Given an input vector compute the prediction of the neural network though a forward pass

        :param input: input vector
        :return: the output/prediction of the neural network
        '''
        alpha = {}
        z = {}
        alpha[0] = input
        for l in range(self.layers):
            z[l] = (np.sum((alpha[l]*self.weights[l]), axis=1) + self.biases[l])[0]
            # Output of neural network (changes the activation function of final layer)
            if l==self.layers-1:
                if self.problem=='regression':
                    alpha[(l+1)] = activation(z[l], 0)
                elif self.problem=='classification':
                    alpha[(l+1)] = activation(z[l], 2)
            else:
                alpha[(l+1)] = activation(z[l], self.active)
        return alpha, z

    def back_propagate(self, alpha, z , output):
        '''
        Compute the output error, back propagate it to previous layers

        :param alpha: current outputs of the activations functions for each layer
        :param z: intermediate parameters
        :param output: expected output of neural network
        :return: errors at each of the nodes
        '''
        delta = {}
        # Error on final output node
        if self.problem=='regression':
            delta[self.layers] = (alpha[self.layers]-output)*activation_derivative(z[self.layers-1], 0)
        elif self.problem=='classification':
            delta[self.layers] = (alpha[self.layers]-output)*activation_derivative(z[self.layers-1], 2)

        # Back propagate output error to previous nodes
        for l in range(self.layers-1,0,-1):
            delta[l] = (np.dot(delta[l+1].T,self.weights[l]))*activation_derivative(z[l-1], self.active)
        return delta

    def gradient(self, alpha, delta, weight_update, bias_update):
        '''
        Compute the gradients for the weight and bias matrices of each layer

        :param alpha: current outputs of the activations functions for each layer
        :param delta: errors at each of the nodes
        :param weight_update: update for the weight variables
        :param bias_update: update for the bias variables
        :return: current gradients of weights and biases
        '''
        for l in range(self.layers,0,-1):
            weight_update[l-1] += np.outer(delta[l],alpha[l-1])
            bias_update[l-1] += delta[l]
        return weight_update, bias_update

    def train(self, nodes, train_x, train_y, test_x, test_y):
        '''
        Train neural network

        :param nodes: number of nodes for each hidden layer
        :param train_x: training examples
        :param train_y: expected outputs of training examples
        :param test_x: test examples
        :param test_y: expected outputs of test examples
        :return loss: output of loss function for each iteration
        '''
        logging.info('Start training')
        loss_=[]
        for i in range(self.iterations):
            outputs, loss = self.apply(test_x, test_y)
            loss_.append(loss/len(test_y))
            logging.info('Iteration %d, loss : %f' %(i, loss/len(test_y)))
            # Initialize the update matrices
            weight_update = {}
            bias_update = {}

            for l in range(self.layers):
                weight_update[l] = np.zeros(shape=(nodes[l], nodes[l + 1])).T
                bias_update[l] = np.zeros(shape=(nodes[l + 1], 1)).T

            # Pass through all data
            for x, y in zip(train_x, train_y):
                alpha, z = self.feed_forward(np.array(x))
                delta = self.back_propagate(alpha, z, y)
                weight_update, bias_update = self.gradient(alpha, delta, weight_update, bias_update)
            # Update weights and biases
            for l in range(self.layers):
                self.weights[l] -= (self.eta/len(train_y))*weight_update[l]
                self.biases[l] -= (self.eta/len(train_y))*bias_update[l]

        return loss_

    def apply(self, test_x, test_y):
        '''
        Apply neural network to test data

        :param test_x: set of test examples
        :param test_y: expected outputs of test examples
        :return: expected outputs, value of loss function
        '''
        nn_outputs = []
        for x in test_x:
            alpha, z = self.feed_forward(x)
            nn_outputs.append(alpha[self.layers])
        loss = loss_function(test_y, nn_outputs, self.problem)
        return np.array(nn_outputs), loss

def loss_function(test_y, estimates, problem):
    '''
    Compute loss function on test data

    :param test_y: ground truth values
    :param estimates: estimated values
    :param problem: type of problem ('regression' or 'classification')
    :return: value of loss function
    '''
    loss = []
    if problem=='regression':
        for t,e in zip(test_y, estimates):
            loss.append((t-e)**2)
        loss = sum(loss)*0.5
    elif problem=='classification':
        for t,e in zip(test_y, estimates):
            loss.append(t*np.log(e)+(1-t)*np.log(1-e))
        loss = -sum(loss)
    return loss

def activation(input, flag):
    '''
    Activation function

    :param input: input to activation function
    :param flag: boolean flag for choosing activation function (0 for identity, 1 for ReLu, 2 for sigmoid)
    :return: output of activation function
    '''
    # Identity function
    if flag==0:
        output=input
    # ReLu
    elif flag==1:
        output = []
        for i in range(len(input)):
            output.append(max(0,input[i]))
    # Sigmoid
    elif flag==2:
        output = []
        for i in range(len(input)):
            output.append(sigmoid(input[i]))
    return np.array(output)

def activation_derivative(input, flag):
    '''
    Derivative of activation function

    :param input: input to derivative of the activation function
    :param flag: boolean flag for choosing activation function (0 for identity, 1 for ReLu)
    :return: output of the derivative of the activation function
    '''
    if flag==0:
        output = [1]*len(input)
    elif flag==1:
        output = []
        for i in range(len(input)):
            if input[i] > 0:
                output.append(1)
            else:
                output.append(0)
    else:
        output=[]
        for i in range(len(input)):
            output.append(sigmoid(input[i])*(1-sigmoid(input[i])))
    return np.array(output)

def sigmoid(x):
    '''
    Sigmoid function

    :param x: input
    :return: output of sigmoid function for x
    '''
    return 1 / (1 + np.exp(-x))

def data_generation_regression(N, lenght_, type):
    '''
    Create length_ number of random N-dimensional data for regression
    (create-shuffle-produce outputs-split)

    :param N: dimensionality of data
    :param lenght_: number of data to create
    :param type: type of function that generates the data
    :return: created data
    '''
    if lenght_<5:
        raise ValueError ("Not enough data to create train and test sets. Try again!")
    # Create random data
    data = np.random.rand(lenght_, N)*2
    np.random.shuffle(data)

    # Linear function
    if type=='linear':
        output = 7*np.sum(data, axis=1) + 3  + np.random.rand(lenght_)

    # Sine function
    elif type=='sine':
        output = [np.math.sin(5*x) for x in np.sum(data, axis=1)]

    # Split data
    train_x, train_y = data[:(int(0.5*len(data)))], output[:(int(0.5*len(data)))]
    test_x, test_y = data[int(0.5*len(data)):], output[int(0.5*len(data)):]
    return train_x, train_y, test_x, test_y

def data_generation_classification(N, length_, margin, separability):
    '''
    Create length_ number of random N-dimensional data for classification (binary)
    (create-shuffle-produce outputs-split)

    :param N: dimensionality of data
    :param lenght_: number of data to create
    :param margin: margin between classes (not separable data margin controls the overlap of classes)
    :param separability: separability of dataset (boolean)
    :return: created data
    '''
    if length_ < 5:
        raise ValueError("Not enough data to create train and test sets. Try again!")
    count_pos = 0
    count_neg = 0
    pos=[]
    neg=[]

    # Separable data
    if separability==True:
        while (count_pos<int(length_/2)) or (count_neg<int(length_/2)):
            point = np.random.rand(N)
            if sum(point)>0.5+margin and count_pos<int(length_/2):
                pos.append(point.tolist())
                count_pos+=1
            elif sum(point)<0.5-margin and count_neg<int(length_/2):
                neg.append(point.tolist())
                count_neg+=1

    # Non-separable data
    else:
        while (count_pos<int(length_/2)) or (count_neg<int(length_/2)):
            point = np.random.rand(N)
            if sum(point)>0.5-margin and count_pos<int(length_/2):
                pos.append(point.tolist())
                count_pos+=1
            elif sum(point)<0.5+margin and count_neg<int(length_/2):
                neg.append(point.tolist())
                count_neg+=1

    # Create labels for data
    data = pos+neg
    output = [0 for i in range(len(neg))]
    output.extend([1 for i in range(len(pos))])

    # Visualize one OR two dimensional data
    if N==1:
        plt.scatter(pos, len(pos) * [1], alpha=0.7, color='blue')
        plt.scatter(neg, len(neg) * [1], alpha=0.7, color='lime')
        plt.show()
    elif N==2:
        for i, d in enumerate(data):
            if output[i] == 0:
                plt.scatter(d[0], d[1], color='blue', alpha=0.7)
            else:
                plt.scatter(d[0], d[1], color='lime', alpha=0.7)
        plt.show()

    return data, output

def analysis(train_x, train_y, test_x, test_y, outputs, loss, problem):
    '''
    Visualizations required for the problem

    :param train_x: input data
    :param train_y: outputs of input data
    :param test_x: test data
    :param test_y: ground truth of
    :param outputs: predicted outputs
    :param loss: loss function values
    :param problem: type of problem ('regression' or 'classification')
    '''

    # Visualize if possible (only for 1D data applied to regression problems)
    if problem=='regression' and train_x.shape[1]==1:
        plt.scatter(train_x, train_y, alpha=0.7, color='blue', label='Training data')
        plt.scatter(test_x, outputs, alpha=1, color='lime', label='Test predictions')
        plt.legend()
        plt.grid()
        plt.show()

    plt.clf()
    plt.xlabel('Iterations')
    plt.ylabel('Loss function')
    plt.plot(loss[1:], color='xkcd:indigo', linewidth=2, marker='.')
    plt.ylim(0)
    plt.xlim(0)
    plt.grid()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Back propagation for fully connected deep neural networks')
    parser.add_argument('no', help='problem number')
    parser.add_argument('letter', help='letter of the sub-problem')
    parser.add_argument('eta', help='update parameter', type=float)
    parser.add_argument('iterations', help='number of iterations', type=int)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if args.no=='1':
        '''
        Configuration for problem 1. (simple network)
        This network contains an input layer and an output layer, with no nonlinearities, while
        a regression loss is used to train the network.
        We test the above configuration using random generated (a) one-dimensional and 
        (b) N-dimensional data. 
        '''

        # (a) 1-dimensional data. Final configuration: N = 500, eta=0.5, iter=10
        if args.letter=='a':
            nodes = [1, 1] # dimensionality of first node controls the dimensionality of the created data
            lenght_ = 500 # number of data to create
            active = 0 # controls whether the ReLu activation is used or not
            train_x, train_y, test_x, test_y = data_generation_regression(nodes[0], lenght_, 'linear')
            nn = Neural_Network(len(nodes)-1, nodes, args.eta, args.iterations, active, 'regression')
            loss = nn.train(nodes, train_x, train_y, test_x, test_y)
            outputs, l = nn.apply(test_x, test_y)
            analysis(train_x, train_y, test_x, test_y, outputs, loss, 'regression')

        # (b) N-dimensional data. Final configuration: N = 500, eta=0.01, iter=20
        elif args.letter=='b':
            loss=[]
            lenght_ = 500
            active = 0
            for n in range(1,12,2):
                nodes = [n, 1]
                train_x, train_y, test_x, test_y = data_generation_regression(nodes[0], lenght_, 'linear')
                nn = Neural_Network(len(nodes) - 1, nodes, args.eta, args.iterations, active, 'regression')
                loss.append(nn.train(nodes, train_x, train_y, test_x, test_y))

            # Plot
            plt.clf()
            for n, l in enumerate(loss):
                plt.plot(l, label=str(2*n+1), linewidth=2.0, marker='.')
            plt.legend()
            plt.grid()
            plt.xlabel('Iterations')
            plt.ylabel('Loss function')
            plt.xlim(xmin=0)
            plt.ylim(ymin=0)
            plt.show()

    elif args.no=='2':
        '''
        Configuration for problem 2. (Shallow network) 
        This is a fully connected neural network with a single hidden layer, and a ReLU non-linearity.
        We test the above configuration using random generated (a) one-dimensional and 
        (b) N-dimensional data. 
        '''

        # (a) 1-dimensional data. Final configuration: N = 1000, eta=0.01, iter=1000, k1=100
        if args.letter=='a':
            nodes = [1, 50, 1]
            lenght_ = 1000
            active = 1
            train_x, train_y, test_x, test_y = data_generation_regression(nodes[0], lenght_, 'sine')
            nn = Neural_Network(len(nodes) - 1, nodes, args.eta, args.iterations, active, 'regression')
            loss = nn.train(nodes, train_x, train_y, test_x, test_y)
            outputs, l = nn.apply(test_x, test_y)
            analysis(train_x, train_y, test_x, test_y, outputs, loss, 'regression')

        # (a) N-dimensional data. Final configuration: N=1000, eta=0.001, iter=20, k1=50
        if args.letter=='b':
            loss=[]
            lenght_ = 1000
            active = 1
            for n in range(1,10,2):
            #for n in range(3, 4):
                nodes = [n, 50, 1]
                train_x, train_y, test_x, test_y = data_generation_regression(nodes[0], lenght_, 'sine')
                nn = Neural_Network(len(nodes) - 1, nodes, args.eta, args.iterations, active, 'regression')
                loss.append(nn.train(nodes, train_x, train_y, test_x, test_y))
            
            # Plot
            plt.clf()
            for n, l in enumerate(loss):
                plt.plot(l, label=str(2*n+1), linewidth=2.0, marker='.')
            plt.legend()
            plt.grid()
            plt.xlabel('Iterations')
            plt.ylabel('Loss function')
            plt.xlim(xmin=0)
            plt.ylim(ymin=0)
            plt.show()

    elif args.no=='3':
        '''
        Configuration for problem 3. (Deep neural network)
        This is a fully-connected network of arbitrary depth.
        '''
        # (a) 1-dimensional data (3 hidden layers). Final configuration: N=1000, eta=0.001, iter=1000
        if args.letter=='a':
            nodes = [1, 20, 10, 20, 1]

            lenght_ = 1000
            active = 1
            train_x, train_y, test_x, test_y = data_generation_regression(nodes[0], lenght_, 'sine')
            nn = Neural_Network(len(nodes) - 1, nodes, args.eta, args.iterations, active, 'regression')
            loss = nn.train(nodes, train_x, train_y, test_x, test_y)
            outputs, l  = nn.apply(test_x, test_y)
            analysis(train_x, train_y, test_x, test_y, outputs, loss, 'regression')

        if args.letter=='b':
            # (a) 1-dimensional data (5 hidden layers). Final configuration: N=1000, eta=0.001, iter=1000
            nodes = [1, 10, 10, 5, 10, 10, 1]
            lenght_ = 1000
            active = 1
            train_x, train_y, test_x, test_y = data_generation_regression(nodes[0], lenght_, 'sine')
            nn = Neural_Network(len(nodes) - 1, nodes, args.eta, args.iterations, active, 'regression')
            loss = nn.train(nodes, train_x, train_y, test_x, test_y)
            outputs, l = nn.apply(test_x, test_y)
            analysis(train_x, train_y, test_x, test_y, outputs, loss, 'regression')

        if args.letter=='c':
            # (a) 2-dimensional data (3 hidden layers). Final configuration: N=1000, eta=0.001, iter=50
            nodes = [2, 20, 10, 20, 1]
            lenght_ = 500
            active = 1
            train_x, train_y, test_x, test_y = data_generation_regression(nodes[0], lenght_, 'sine')
            nn = Neural_Network(len(nodes) - 1, nodes, args.eta, args.iterations, active, 'regression')
            loss = nn.train(nodes, train_x, train_y, test_x, test_y)
            outputs, l = nn.apply(test_x, test_y)
            analysis(train_x, train_y, test_x, test_y, outputs, loss, 'regression')

        if args.letter=='d':
            # (b) 2-dimensional data (5 hidden layers). Final configuration: N=1000, eta=0.001, iter=50
            nodes = [2, 10, 10, 5, 10, 10, 1]
            lenght_ = 1000
            active = 1
            train_x, train_y, test_x, test_y = data_generation_regression(nodes[0], lenght_, 'sine')
            nn = Neural_Network(len(nodes) - 1, nodes, args.eta, args.iterations, active, 'regression')
            loss = nn.train(nodes, train_x, train_y, test_x, test_y)
            outputs, l = nn.apply(test_x, test_y)
            analysis(train_x, train_y, test_x, test_y, outputs, loss, 'regression')

        if args.letter=='e':
            # Depth VS Convergence. Final configuration: N=1000, eta=0.001, iter=100
            loss = []
            lenght_ = 1000
            active = 1
            train_x, train_y, test_x, test_y = data_generation_regression(1, lenght_, 'sine')

            n = [ [1,10,10,1], [1,7,7,7,1]]
            for nodes in n:
                nn = Neural_Network(len(nodes) - 1, nodes, args.eta, args.iterations, active, 'regression')
                loss.append(nn.train(nodes, train_x, train_y, test_x, test_y))

            # Plot
            plt.clf()
            for n, l in enumerate(loss):
                plt.plot(l, label=str(n+2)+' '+str('hidden layers'), linewidth=2.0, marker='.')
            plt.legend()
            plt.grid()
            plt.xlabel('Iterations')
            plt.ylabel('Loss function')
            plt.xlim(xmin=0)
            plt.ylim(ymin=0)
            plt.show()

    elif args.no=='4':
        '''
        Binary Classification problem, for linear and non-linear separable 1D and ND data.
        Problem configurations: step=1, iter=1000
        '''

        # Linear separable 1D data
        if args.letter=='a' or args.letter=='b':
            if args.letter=='a':
                nodes = [1, 1]
            elif args.letter=='b':
                nodes = [2, 1]
            m = [0.1, 0.05, 0.01, 0.001]
            lenght_ = 500
            active = 0
            loss = []

            for margin in m:
                data, output = data_generation_classification(nodes[0], lenght_, margin, True)
                train_x, train_y = data[::2], output[::2]
                test_x, test_y =  data[1::2], output[1::2]
                nn = Neural_Network(len(nodes) - 1, nodes, args.eta, args.iterations, active, 'classification')
                loss.append(nn.train(nodes, train_x, train_y, test_x, test_y))

            # Plot
            plt.clf()
            for margin, l in zip(m,loss):
                plt.plot(l, label=margin, linewidth=2.0, marker='.')
            plt.legend()
            plt.grid()
            plt.xlabel('Iterations')
            plt.ylabel('Loss function')
            plt.xlim(xmin=0)
            plt.ylim(ymin=0)
            plt.show()

        # Non-linear separable
        if args.letter=='c' or args.letter=='d':
            if args.letter=='c':
                nodes = [1, 1]
            elif args.letter=='d':
                nodes = [2, 1]
            margin = 0.1 # margin here controls the overlap between non-linear separable classes
            lenght_ = 500
            active = 0

            data, output = data_generation_classification(nodes[0], lenght_, margin, False)
            train_x, train_y = data[::2], output[::2]
            test_x, test_y = data[1::2], output[1::2]
            nn = Neural_Network(len(nodes) - 1, nodes, args.eta, args.iterations, active, 'classification')
            loss = nn.train(nodes, train_x, train_y, test_x, test_y)
            outputs, l = nn.apply(test_x, test_y)
            analysis(train_x, train_y, test_x, test_y, outputs, loss, 'classification')

if __name__ == '__main__':
    main()
