This readme file contains information on how to produce each of the results presented
in my report. The choice of the hyper-parameters was made after tuning. Plots are created
 using plt.show() so you'd have to close them in order to continue the execution of the code.

____________________________________________________________________________________________

>> ADULT DATASET
____________________________________________________________________________________________

> To implement the configuration 67 -> 15 -> 1 run:

python3 adult.py True --verbose

> To implement the configuration 67 -> 30 -> 15 -> 1 run:

python3 adult.py False --verbose

___________________________________________________________________________________________

>> FLOWERS
___________________________________________________________________________________________

python3 flowers.py --verbose

Note: The analysis part does not run by default. If  you want to run the analysis part
 (weights print outs and misclassified examples) set --analysis argument equal to True.

_________________________________________________________________________________________

>> THREE METER
________________________________________________________________________________________

python3 three_meter.py --verbose

Note: The analysis part does not run by default. If  you want to run the analysis part
 (most poorly fitted examples from training set) set --analysis argument equal to True.  
