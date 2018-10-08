
This readme file contains information on how to produce each
of the results presented in my report. The choice of the
hyper-parameters was made after tuning. Plots are created 
using plt.show() so you'd have to close them in order to 
continue the execution of the code.

* Note: For the best configurations you could use either 
the argument 'simple' which implements an architecture with
one convolutional layers, or 'second' which adds a second
convolutional layer.

Best configurations:
------------------------------------------------------

python3 mnist_nn.py final simple --verbose

python3 fashion_mnist_nn.py final simple --verbose

python3 breast_cancer.py final predefined --verbose

> OR (add a second conv layer)

python3 mnist_nn.py final second --verbose

python3 fashion_mnist_nn.py final second --verbose


------------------------------------------------------
Tuning experiments:
------------------------------------------------------
> MNIST:

python3 mnist_nn.py tuning filter --verbose

python3 mnist_nn.py tuning pooling --verbose

python3 mnist_nn.py tuning dropout --verbose

python3 mnist_nn.py tuning num_filter --verbose

> FMINST: 

python3 fashion_mnist_nn.py tuning filter --verbose

python3 fashion_mnist_nn.py tuning pooling --verbose

python3 fashion_mnist_nn.py tuning dropout --verbose

python3 fashion_mnist_nn.py tuning num_filter --verbose


> Breast Cancer:

python3 breast_cancer.py tuning batch_size --verbose

python3 breast_cancer.py tuning nodes --verbose

python3 breast_cancer.py tuning layers --verbose




