This readme file contains information on how to produce each
of the results presented in my report. The choice of the 
hyper-parameters was made after tuning. Plots are created using
plt.show() so you'd have to close them in order to continue 
the execution of the code. 
---------------------------------------------------------------
python3 back_propagation.py 1 a 0.5 500 --verbose
python3 back_propagation.py 1 b 0.01 20 --verbose
---------------------------------------------------------------
python3 back_propagation.py 2 a 0.01 1000 --verbose
python3 back_propagation.py 2 b 0.001 20 --verbose
---------------------------------------------------------------
python3 back_propagation.py 3 a 0.001 5000 --verbose
python3 back_propagation.py 3 b 0.001 5000 --verbose
python3 back_propagation.py 3 c 0.001 50 --verbose
python3 back_propagation.py 3 d 0.001 50 --verbose
python3 back_propagation.py 3 e 0.001 100 --verbose
---------------------------------------------------------------
python3 back_propagation.py 4 a 1 1000 --verbose
python3 back_propagation.py 4 b 1 1000 --verbose
python3 back_propagation.py 4 c 1 1000 --verbose
python3 back_propagation.py 4 d 1 1000 --verbose
---------------------------------------------------------------


* Note: For problem that deal with data that are randomly 
created you would probably get slightly different results
each time you run the code. 