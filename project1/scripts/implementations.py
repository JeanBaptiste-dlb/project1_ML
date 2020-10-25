import numpy as np
import numpy.linalg as lin

from proj1_helpers import *

# Base Functions


def compute_MSE_loss(y, tx, w):
    """Calculate the MSE loss.
        Parameters :
            y (numpy.ndarray): An array with shape (n,1)
            tx (numpy.ndarray): An array with shape (n,m)
            w (numpy.ndarray): An array with shape (k,1)
        Returns:
            loss (numpy.float64): the MSE loss
    """
    return (1/(2*len(y))) * np.sum((y-tx.dot(w))**2)


def compute_gradient(y, tx, w):
    """Compute the gradient.
            Parameters:
                    y (numpy.ndarray): An array with shape (n,1)
                    tx (numpy.ndarray): An array with shape (n,m)
                    w (numpy.ndarray): An array with shape (m,1) describing the model
            Returns:
                    grad (numpy.ndarray): An array with shape (m,1), the gradient
    """

    ew = y-tx.dot(w.T)
    grad = -(1/len(y))*(tx.T.dot(ew))
    return grad


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels.
            Parameters:
                    y (numpy.ndarray): An array with shape (n,1)
                    tx (numpy.ndarray): An array with shape (n,m)
                    w (numpy.ndarray): An array with shape (m,1) describing the model
            Returns:
                    grad (numpy.ndarray): An array with shape (m,1), the gradiant
    """
    ew = y-tx.dot(w.T)
    grad = -(1/len(y))*(tx.T.dot(ew))
    return grad


def sigmoid(t):
    """apply the sigmoid function on t.
            Parameters:
                    t (Union[numeric, numpy.ndarray]) 
            Returns:
                    (Union[numeric,  numpy.ndarray])
    """
    return np.exp(t)/(1+np.exp(t))


def calculate_loss(y, tx, w, lambda_=0):
    """compute the loss: negative log likelihood. divided by 2*n to have more readable values.
            Parameters:
                    y (numpy.ndarray): An array with shape (n,1)
                    tx (numpy.ndarray): An array with shape (n,m)
                    w (numpy.ndarray): An array with shape (m,1) describing the model
            Returns:
                    sum (float) : the loss : negative log likelihood
    """
    loss = 0
    txw = tx.dot(w)
    for i in range(len(y)):
        loss += - y[i]*txw[i] + np.log(1 + np.exp(txw[i]))
        loss += lambda_ * np.linalg.norm(w)**2
    return loss/(2*len(y))


def stats(y, y_pred):
    """compute the confusion matrix for the y_pred regarding y.

    Args:
        y (numpy.ndarray): labels
        y_pred (np.ndarray): predictions

    Returns:
        (int, int, int,float, float): #True Positives,  #False Positives, #False Negatives, precision, recall
    """
    TP = len([i for i, j in zip(y_pred, y) if i == j == 1])
    #TN = len([i for i, j in zip(y_pred, y) if i == j != 1])
    FP = len([i for i, j in zip(y_pred, y) if i != j and i == 1])
    FN = len([i for i, j in zip(y_pred, y) if i != j and i != 1])

    if(TP+FP == 0):
        precision = -1
    else:
        precision = TP / (TP + FP)
    if(TP+FN == 0):
        recall = -1
    else:
        recall = TP / (TP + FN)

    return TP, FP, FN, precision, recall


def f1_score(y, y_pred):
    """compute the F1_score for y and y_pred. 
    Args:
        y (np.ndarray): labels  
        y_pred (np.ndarray): predictions
    Returns:
        float: the F1 score of the y_pred regarding y.
    """

    precision = stats(y, y_pred)[3]
    recall = stats(y, y_pred)[4]
    # We give the same F1 score (0) as the grading platform if our model does not predict anything at all.
    if(precision == -1 or recall == -1):
        return 0

    return (2*precision*recall)/(precision+recall)


def calculate_gradient(y, tx, w, lambda_=0):
    """compute the gradient of loss.
            Parameters:
                    y (numpy.ndarray): An array with shape (n,1)
                    tx (numpy.ndarray): An array with shape (n,m)
                    w (numpy.ndarray): An array with shape (m,1) describing the model
            Returns:
                    grad (numpy.ndarray): An array with shape (m,1), the gradient
    """
    inner = sigmoid(tx.dot(w))-y
    grad = tx.T.dot(inner) + 2*lambda_*w
    return grad

# ! TODO refactor names and polynomial regression


def yy2(tx, k, ld):

    N, n = tx.shape
    tx2 = []
    y = []
    for i in range(N):
        z = tx[i, k]
        if z != -999:
            l = tx[i]
            tx2 += [np.delete(l, ld)]
            y += [z]
    return np.array(tx2), np.array(y)


def yy3(tx, ld):
    N, n = tx.shape
    tx3 = []
    for i in range(N):
        l2 = tx[i]
        tx3 += [np.delete(l2, ld)]
    return np.array(tx3)


def Datas_completion_lacking_values_predicted(tx):
    """complete tx using a regression
    Args:
        tx (numpy.ndarray): with shape(n,m)

    Returns:
        numpy?ndarray: with shape (n,m)
    """
    N, n = tx.shape
    # columns with lacking values (-999)
    l = [0, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28]
    tx2 = np.zeros((N, n))
    for i in range(n):
        if i not in l:
            for j in range(N):
                tx2[j, i] = tx[j, i]
    txk2 = yy3(tx, l)
    for k in l:
        txk, yk = yy2(tx, k, l)
        w, loss = get_w_loss(yk, txk, 3)
        yreg = np.dot(txk2, w)
        for j in range(N):
            if tx[j, k] == -999:
                tx2[j, k] = yreg[j]
            else:
                tx2[j, k] = tx[j, k]
    return tx2


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def build_k_indices(n, k_fold, seed=0):
    """genrates indices for k-fold

    Args:
        n (int): number of indices
        k_fold (int): number of indices/folds to generate. should be unsigned.
        seed (integer): seed to use for pseudo-random indices generation, a seed value is set by default for reproducibility. Defaults to 0.

    Returns:
        np.ndarray: An array with shape (k_fold,n/k_fold) containing k_fold list of shuffled indices. Concatenated, the folds form a shuffled range(0, n). 
    """
    interval = int(n / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(n)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_split(y, x, k_indices, k):
    """generate a train test split with k_indices as the split, k-th subset as the test set.
    Args:
        y (numpy.ndarray): An array with shape (n,1)
        x (numpy.ndarray): An array with shape (n,m)
        k_indices (numpy.ndarray): An array with shape (number_of_folds, n/number_of_folds)
        k (int): should be in [0,..., number_of_folds]. uses the k_th subset as the test set.
    Returns:
        (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray): returns x_train, y_train, x_test, y_test with respective shapes (n,m-(n/number_of_folds)), (n-(n/number_of_folds),1), (n,n/number_of_folds), (n/number_of_folds, 1)

    """
    # get k'th subgroup in test, others in train:
    test = k_indices[k]
    k_1 = k_indices[:k].reshape(1, len(k_indices[:k])*len(k_indices[0]))
    k_2 = k_indices[(k+1):].reshape(1, len(k_indices[(k+1):])
                                    * len(k_indices[0]))
    train = np.concatenate([k_1[0], k_2[0]])
    return x[[train]], y[[train]], x[[test]], y[[test]]


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x (numpy.ndarray): An array of shape (n,m) to expend.
        degree (int): should be unsigned

    Returns:
        numpy.ndarray: Expended x with shape (n, degree+1, m)
    """

    basis = []
    for row in x:
        vec = []
        for j in range(1, degree+1):
            vec += list(map(lambda x: pow(x, j), row))
        vec.append(1)
        basis.append(vec)
    return np.array(basis)


def normalize(x):
    """normalize each column of the dataset using Min-Max feature scaling.

    Args:
        x (numpy.ndarray): An array of shape (n,m) 

    Returns:
        numpy.ndarray: tx normalized.
    """
    tmp = []
    for column in x.T:
        min = column.min()
        max = column.max()
        new_col = (column - min)/(max-min)
        tmp.append(new_col)
    return np.array(tmp).T


# Algorithms

# least squares gradient descent

def least_squares_GD(y, tx, initial_w, max_iters, gamma, log=False, store=False):
    """Compute the Gradiant Descent with least squares loss function.
            Parameters:
                    y (numpy.ndarray): An array with shape (n,1)
                    tx (numpy.ndarray): An array with shape (n,m)
                    initial_w (Union[float, numpy.ndarray]): An array with shape (m,1) describing the model or a float which will fill an array with shape (m,1)
                    max_iters (int) : number of iterations
                    gamma (numpy.float64): gradiant multiplier gamma > 0 
                    log (bool): if set to True display step by step informations about the descent
            Returns:
                    (numpy.ndarray, Union[numpy.ndarray, float]): [0] w : ndarray with shape (n_iters,m) if store, else shape(m,1), [1] loss: ndarray with shape (n_iters,) if store, else float

    """
    # Define parameters to store w and loss
    if isinstance(initial_w, np.ndarray):
        ws = [initial_w]
        w = initial_w
    else:
        w = np.array([initial_w]*tx.shape[1])
        ws = [w]
    losses = []
    loss = 0
    for n_iter in range(max_iters):
        # compute gradient, loss and next iter w
        grad = compute_gradient(y, tx, w)
        loss = compute_MSE_loss(y, tx, w)
        w = (-gamma*grad).T+w
        # store w and loss
        if store:
            ws.append(w)
            losses.append(loss)
        if(log):
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    if (store):
        return ws, losses
    else:
        return w, loss


# stochastic gradient descent

def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, batch_size=1, log=False, store=False):
    """Compute the Stochastic Gradiant Descent with least squares loss function.
            Parameters:
                    y (numpy.ndarray): An array with shape (n,1)
                    tx (numpy.ndarray): An array with shape (n,m)
                    initial_w (Union[float, numpy.ndarray]): An array with shape (m,1) describing the model or a float which will fill an array with shape (m,1)
                    max_iters (int) : number of iterations
                    gamma (numpy.float64): gradiant multiplier gamma > 0 
                    log (bool): if set to True display step by step informations about the descent
                    batch_size (int): default to 1, number of data_points used to compute the gradient  
                    log (bool): if set to True display step by step informations about the descent. Defaults to False.
            Returns:
                   (numpy.ndarray, Union[numpy.ndarray, float]): [0] w : ndarray with shape (n_iters,m) if store, else shape(m,1), [1] loss: ndarray with shape (n_iters,) if store, else float
    """
    iterator = batch_iter(
        y, tx, batch_size, num_batches=max_iters, shuffle=True)
    if isinstance(initial_w, np.ndarray):
        ws = [initial_w]
        w = initial_w
    else:
        w = np.array([initial_w]*tx.shape[1])
        ws = [w]
    losses = []
    for y, x in iterator:
        # compute gradient and loss
        grad = compute_stoch_gradient(y, x, w)
        loss = compute_MSE_loss(y, x, w)
        # update w by gradient
        w = (-gamma*grad).T+w
        # store w and loss
        if (store):
            ws.append(w)
            losses.append(loss)
        if(log):
            print("Gradient Descent(): loss={l}, w0={w0}, w1={w1}".format(
                l=loss, w0=w[0], w1=w[1]))
    if store:
        return ws, losses
    else:
        return w, loss


# least squares

def least_squares(y, tx):
    """calculate the least squares solution.
            Parameters:
                    y (numpy.ndarray): An array with shape (n,1)
                    tx (numpy.ndarray): An array with shape (n,m)
            Returns:
                    (numpy.ndarray, float): [0] w*: An array with shape (m,1) the optimal model for complexity m, [1] loss: MSE loss 
    """
    xtx = tx.T.dot(tx)
    a = lin.inv(xtx)
    w = a.dot(tx.T).dot(y)
    loss = compute_MSE_loss(y, tx, w)

    return w, loss


# ridge regression

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.
            Parameters:
                    y (numpy.ndarray): An array with shape (n,1)
                    tx (numpy.ndarray): An array with shape (n,m)
                    lambda_ (numpy.float64): regularization factor lambda_>0
            Returns:
                    (numpy.ndarray, float): [0] w*: An array with shape (m,1) the optimal model for complexity m and lambda_ regularization factor, [1] loss: MSE loss 
    """
    lambda_p = 2*len(y)*lambda_
    w = lin.inv(tx.T.dot(tx)+lambda_p*np.identity(len(tx[0]))).dot(tx.T).dot(y)
    loss = (compute_MSE_loss(y, tx, w))
    return w, loss
# Logistic regression


def learning_by_gradient_descent(y, tx, w, gamma, lambda_=0):
    """Do one step of gradient descent using logistic regression. Returns the loss and the updated w.

    Args:
        y (numpy.ndarray): An array with shape (n,1)
        tx (numpy.ndarray): An array with shape (n,m)
        w (numpy.ndarray): an array with shape (m, 1)
        gamma (float): descent scale factor, should be positive.
    Returns:
        (numpy.ndarray, float): [0] w: ndarray with shape(m,1), [1] loss: MSE loss
    """
    loss = calculate_loss(y, tx, w, lambda_)
    grad = calculate_gradient(y, tx, w, lambda_)
    w = w - (grad * gamma)
    return w, loss

# logistic regression without regularization


def logistic_regression_gradient_descent(y, tx, initial_w, max_iters, gamma, threshold=1e-15, log=False, store=False):
    """Logistic regression using gradient descent

    Args:
        y (numpy.ndarray): An array with shape (n,1)
        tx (numpy.ndarray): An array with shape (n,m)
        initial_w (Union[float, numpy.ndarray], optional): An array with shape (m,1) describing the model. Defaults to 0.
        max_iters (int, optional): number of iterations. Defaults to 1000.
        gamma (float, optional): descent scale factor, should be positive. Defaults to 0.01.
        threshold (float, optional): factor allowing the algorithm to stop in case of linear separability of the datapoints. Defaults to 1e-8.
        log (bool): if set to True display step by step informations about the descent. Defaults to False.
        store (bool, optional): if True stores all the gradients and loss from the descent. Defaults to False.
    Returns:
        (numpy.ndarray, Union[numpy.ndarray, float]): [0] w : ndarray with shape (n_iters,m) if store, else shape(m,1), [1] loss: ndarray with shape (n_iters,) if store, else float
    """
    # init parameters
    loss = np.inf
    last_loss = np.inf
    losses = []
    ws = []
    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), tx]
    if isinstance(initial_w, np.ndarray):
        ws = [initial_w]
        w = initial_w
    else:
        w = np.array([initial_w]*tx.shape[1])
        ws = [w]
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        last_loss = loss
        w, loss = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if log and iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        if store:
            losses.append(loss)
            ws.append(w)
        # converge criterion
        if np.abs(last_loss-loss) < threshold:
            print("threshold exit")
            break
    # visualization
    if store:
        return ws, losses
    else:
        return w, loss


# Logistic regression with regularization

def reg_logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_, threshold=1e-15, log=False, store=False):
    """Logistic regression using gradient descent

    Args:
        y (numpy.ndarray): An array with shape (n,1)
        tx (numpy.ndarray): An array with shape (n,m)
        initial_w (Union[float, numpy.ndarray], optional): An array with shape (m,1) describing the model. Defaults to 0.
        max_iters (int, optional): number of iterations. Defaults to 1000.
        gamma (float, optional): descent scale factor, should be positive. Defaults to 0.01.
        threshold (float, optional): factor allowing the algorithm to stop in case of linear separability of the datapoints. Defaults to 1e-8.
        log (bool, optional): if True print 100 step by 100 step logs for the descent. Defaults to False.
        store (bool, optional): if True stores all the gradients and losses from the descent. Defaults to False.
        lambda (float, optional): scale factor for the regularization, if to high risk of under fitting , if to low risk of overfiting.Defaults to 0.1.
    Returns:
        (numpy.ndarray, Union[numpy.ndarray, float]): [0] w : ndarray with shape (n_iters,m) if store, else shape(m,1), [1] loss: ndarray with shape (n_iters,) if store, else float
    """
    # init parameters
    loss = np.inf
    last_loss = np.inf
    losses = []
    ws = []
    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), tx]
    if isinstance(initial_w, np.ndarray):
        ws = [initial_w]
        w = initial_w
    else:
        w = np.array([initial_w]*tx.shape[1])
        ws = [w]
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        last_loss = loss
        w, loss = learning_by_gradient_descent(y, tx, w, gamma, lambda_)
        # log info
        if log and iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        if store:
            losses.append(loss)
            ws.append(w)
        # converge criterion
        if np.abs(last_loss-loss) < threshold:
            print("threshold exit")
            break
    # visualization
    if store:
        return ws, losses
    else:
        return w, loss


# apply method
def get_w_loss(y, x, method, initial_w=0, max_iters=10, gamma=0.00000005, threshold=1e-08, logs=False, batch_size=1, lambda_=0.005, store=False):
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
    if(method == 1):
        return least_squares_GD(y, x, initial_w, max_iters, gamma, logs, store)
    elif(method == 2):
        return stochastic_gradient_descent(y, x, initial_w, max_iters, gamma, batch_size, logs, store)
    elif(method == 3):
        return least_squares(y, x)
    elif(method == 4):
        return ridge_regression(y, x, lambda_=lambda_)
    elif(method == 5):
        return logistic_regression_gradient_descent(y, x, initial_w, max_iters, gamma, threshold, logs, store)
    elif(method == 6):
        return reg_logistic_regression(y, x, initial_w, max_iters, gamma, threshold, logs, store, lambda_)
    else:
        return ValueError


def submit(_, tX_test, w, ids_test, method, degree):
    """create a submission for the model w. 

    Args:
        method (int): method used to generate w :   1 -> least_squares_GD               
                                                    2 -> stochastic_gradient_descent
                                                    3 -> least_squares
                                                    4 -> ridge_regression
                                                    5 -> logistic_regression_gradient_descent
                                                    6 -> reg_logistic_regression
        degree (int): degree of the feature expension used to generate w.
        w (np.ndarray): model for the submission
    """
    OUTPUT_PATH = "../data/submission.csv"
    if(method in [5, 6]):
        data = np.c_[np.ones((_.shape[0], 1)), build_poly(tX_test, degree)]
    else:
        data = build_poly(tX_test, degree)
    submission_y = predict_labels(w, data)
    create_csv_submission(ids_test, submission_y, OUTPUT_PATH)
    return None
