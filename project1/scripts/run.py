# iimports 

import importlib.util
import implementations
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *

#load datas 
DATA_TRAIN_PATH = "../data/train.csv" 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

DATA_TEST_PATH = "../data/test.csv" 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# normalize datas
def normalize (tx):
    tmp=[]
    for column in tx.T:
        min = column.min()
        max = column.max()
        new_col=(column - min)/(max-min)
        tmp.append(new_col)
    return np.array(tmp).T

tX_test=normalize(tX_test)
tX=normalize(tX)
y[y==-1]=0

#apply method
def get_w_loss(y,x,method,initial_w=0,max_iters=10,gamma=0.00000005,threshold=1e-08,logs=False,batch_size=1,lambda_=0.005, store=False):
    """applies the algorithm described by <method>. by default as the awaited behaviour for the project, but offers some possible enhancements (batch_size,threshold, logs, store)
    Args:
        y (numpy.ndarray): An array with shape (n,1)
        tx (numpy.ndarray): An array with shape (n,m)
        method (int): 1 -> least_squares_GD*                
                      2 -> stochastic_gradient_descent*
                      3 -> least_squares
                      4 -> ridge_regression
                      5 -> logistic_regression_gradient_descent*
                      6 -> reg_logistic_regression*
        * this algorithm is an iterative descent
        initial_w (Union[float, numpy.ndarray], optional): if a int is set w will start with an array. Defaults to 0.
        max_iters (int, optional): number of iterations (only for descent algorithms). Defaults to 10.
        gamma (float, optional): descent scale factor (only for descent algorithms). Defaults to 0.00000005.
        threshold (float, optional): threshold to prevent divergence in case of linear separability (I assume). Defaults to 1e-08.
        log (bool, optional): if True step by step logs are displayed (only for descent algorithms). Defaults to False.
        batch_size (int, optional): size of the sample on which to compute the partial gradient for stochastic gradient descent algorithm. Defaults to 1.
        lambda_ (float, optional): regularization factor with good values allows to limit overfitting and underfitting (only for ridge_regression and reg_logistic_regression). Defaults to 0.005.
        store (bool, optional): [description]. Defaults to False.
    Returns:
        w (numpy.ndarray): ndarray with shape (n_iters,m) if store and if the method is a descent, else shape(m,1) 
        loss (Union[numpy.ndarray, float]): ndarray with shape (n_iters,) if store and if the method is a descent, else float
    :raises ValueError: if method is not <6
    """
    initial_=np.array([initial_w]*len(x[0]))
    if(method==1):
        return implementations.least_squares_GD(y,x, initial_, max_iters, gamma, logs)
    elif(method==2): 
        return implementations.stochastic_gradient_descent(y,x, initial_, max_iters, gamma, batch_size, logs)
    elif(method==3):
        return implementations.least_squares(y,x)
    elif(method==4):
        return implementations.ridge_regression(y,x, lambda_=lambda_)
    elif(method==5):
        return implementations.logistic_regression_gradient_descent(y,x,initial_w,max_iters,gamma,threshold,logs,store)
    elif(method==6):
        return implementations.reg_logistic_regression(y,x,initial_w,max_iters,gamma,threshold,logs,store, lambda_)
    else:
        return ValueError
    
OUTPUT_PATH = '../data/submission.csv'
#for i in range(-10,10,1):
#    ml,weights=get_w(y,tX,2,i,max_iters=1000,log=False)
#    mls.append(ml)
#    weightss.append(weights)
#print(np.where(mls==min(mls)))
#print(mls)
method=6

weights, loss=get_w_loss(y,tX,method,logs=True,max_iters=10000, gamma=0.0000005)
print(str(weights) + str(loss))
#weights=weightss[np.where(mls==min(mls))[0][0]]

if(method==5 or method==6):
    data = np.c_[np.ones((_.shape[0], 1)), tX_test]
else:
    data=tX_test
y_pred = predict_labels(weights, data)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)