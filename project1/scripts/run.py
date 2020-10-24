
# imports 
import importlib.util
import implementations as implementations
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
#load datas 
import csv
import numpy as np

DATA_TRAIN_PATH = "../data/train.csv"
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

DATA_TEST_PATH = "../data/test.csv" 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

OUTPUT_PATH = '../data/submission.csv'

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

    
def I_do_it_all_and_I_try_to_do_it_good_REG_LOG_REG( degree, lambdas_,k_fold=10):
    method=6
    indices=implementations.build_k_indices(len(y),k_fold)
    enhanced_tX = implementations.build_poly(tX, degree)
    best_heuristique= best_accuracy= best_TP= best_TS= best_lambda= best_losses_tr= best_losses_te=0
    best_w=[]
    print("1 : Train test split")
    x_train, y_train, x_test, y_test=implementations.cross_validation_split(y,enhanced_tX,indices,0)
    print("2 : Compute regularized logistic regression")
    for lambda_ in lambdas_:
        w, loss_tr = get_w_loss(y_train,x_train,method,gamma=0.00005,max_iters=100,lambda_=0.001)
        x_test = np.c_[np.ones((y_test.shape[0], 1)), x_test]
        print("3 : Predict using generated model with {}".format(lambda_))
        y_pred = predict_labels(w,x_test)
        y_test[y_test==0]=-1
        print("4 : Compute stats with {}".format(lambda_))
        matches = [i for i, j in zip(y_pred, y_test) if i == j]  
        accuracy = len(matches)/len(y_test)
        F1 = implementations.f1_score(y_test,y_pred)
        if (2*accuracy + F1 > best_heuristique):  # As the set seems not to much unbalanced I give more importance to accuracy than F1.
            best_w = w
            best_TP, best_TN, best_FP, best_FN = implementations.stats(y_test,y_pred)
            best_lambda= lambda_
            best_F1 = F1
            best_accuracy = accuracy
            loss_te_best = loss_te=implementations.calculate_loss(y_test,x_test,w,lambda_)
            loss_tr_best=loss_tr
    print("5 : Generate the submission")
    implementations.submit(_,tX_test,best_model,ids_test,method,degree)
    return best_model, losses_tr_best, losses_te_best, best_lambda, best_accuracy, best_F1, best_TP, best_TN, best_FP, best_FN

def trial_reg_log_reg():
    '''
        trial with regularized logistic regression
    '''
    method=6 
    if(method==5 or method==6):
        data = np.c_[np.ones((_.shape[0], 1)), tX_test]
    else:
        data=tX_test
    gamma=5/2500000
    weights, losses = implementations.get_w_loss(y,tX,method,1,max_iters=50,gamma=gamma, lambda_=0.001,store=True)

    norms = list(map(np.linalg.norm, weights))
 
    index = np.where(norms == min(norms))[0][0]
    y_pred = implementations.predict_labels(weights[index], data)
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
    fig, axs = plt.subplots(2,1,facecolor=(.18, .31, .31))
    axs[0].plot(losses)
    axs[1].plot([np.linalg.norm(item) for item in weights])
    axs[0].set_facecolor('#eafff5')
    axs[1].set_facecolor('#eafff5')
    axs[0].set_title('evolution of loss with iterations reg log regression', color='0.7')
    axs[0].set_ylabel("loss", color='tab:orange')
    axs[1].set_ylabel("complexity", color='tab:orange')
    axs[1].set_xlabel("iter",color='tab:orange')
    axs[0].tick_params(labelcolor='tab:orange')
    axs[1].tick_params(labelcolor='tab:orange')
    plt.savefig("../visualisations/trial_reg_log_regression.png")
    plt.tight_layout()
    plt.show()
    return None

 

def trial_cat_531_F1_536():
    '''
        trial with least squares : 
        categorical accuracy : 0.531
        F1 : 0.536
    '''
    method=3
    if(method==5 or method==6):
        data_test = np.c_[np.ones((_.shape[0], 1)), tX_test]
    else:
        data_test=tX_test
    weights, losses = get_w_loss(y,tX,method,store=True)
    y_pred = predict_labels(weights, data_test)
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
    return None

lambdas_=list(map(lambda x: 1/(10**(x[0]+1)), enumerate([1]*10)))
best_model, losses_tr_best, losses_te_best, best_lambda, best_accuracy, best_F1, best_TP, best_TN, best_FP, best_FN = I_do_it_all_and_I_try_to_do_it_good_REG_LOG_REG(9, lambdas_)
