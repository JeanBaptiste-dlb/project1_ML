import numpy as np
import numpy.linalg as lin

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
                    t (numeric | numpy.ndarray)
            Returns:
                    sigmoid (numeric | numpy.ndarray)
    """
    return np.exp(t)/(1+np.exp(t))


def calculate_loss(y, tx, w, lambda_=0):
    """compute the loss: negative log likelihood.
            Parameters:
                    y (numpy.ndarray): An array with shape (n,1)
                    tx (numpy.ndarray): An array with shape (n,m)
                    w (numpy.ndarray): An array with shape (m,1) describing the model
            Returns:
                    sum (float) : the loss : negative log likelihood
    """
    loss = 0
    txw=tx.dot(w)
    for i in range(len(y)):
        loss+= - y[i]*txw[i] + np.log(1 + np.exp(txw[i]))
        loss+= lambda_ * np.linalg.norm(w)**2

    #for i in range(len(y)):
    #    sum += np.log(1+np.exp(tx[i].T.dot(w)))
    #    sum -= y[i]*(tx[i].T.dot(w))
    # sum=np.sum(np.log(1+np.exp(tx.T.dot(w))-y.dot(tx.T.dot(w))))
    return loss


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
    grad= tx.T.dot(inner) + 2*lambda_*w
    return grad


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
                    w (numpy.ndarray): ndarray with shape (n_iters,m) if store, else shape(m,1)
                    loss (Union[numpy.ndarray, float]): ndarray with shape (n_iters,) if store, else float
    """
    # Define parameters to store w and loss
    if type(initial_w) == np.ndarray:
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
    if store:
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
                    w (numpy.ndarray): ndarray with shape (n_iters,m) if store, else shape(m,1)
                    loss (Union[numpy.ndarray, float]): ndarray with shape (n_iters,) if store, else float
    """
    iterator = batch_iter( y, tx, batch_size, num_batches=max_iters, shuffle=True)
    if type(initial_w) == np.ndarray:
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
                    w* (numpy.ndarray): An array with shape (m,1) the optimal model of complexity k
                    loss (float): MSE loss
    """
    w = lin.inv(tx.T.dot(tx)).dot(tx.T).dot(y)
    loss= compute_MSE_loss(y, tx, w)
    
    return w, loss


# ridge regression

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.
            Parameters:
                    y (numpy.ndarray): An array with shape (n,1)
                    tx (numpy.ndarray): An array with shape (n,m)
                    lambda_ (numpy.float64): regularization factor lambda_>0
            Returns:
                    w* (numpy.ndarray): An array with shape (m,1) the optimal model
                    loss (float): MSE loss
    """
    lambda_p = 2*len(y)*lambda_
    w = lin.inv(tx.T.dot(tx)+lambda_p*np.identity(len(tx[0]))).dot(tx.T).dot(y)
    loss = (compute_MSE_loss(y, tx, w))
    return w, loss
# Logistic regression


def learning_by_gradient_descent(y, tx, w, gamma,lambda_=0):
    """Do one step of gradient descent using logistic regression. Returns the loss and the updated w.

    Args:
        y (numpy.ndarray): An array with shape (n,1)
        tx (numpy.ndarray): An array with shape (n,m)
        w (numpy.ndarray): an array with shape (m, 1)
        gamma (float): descent scale factor, should be positive.
    Returns:
        w (numpy.ndarray): ndarray with shape(m,1) 
        loss (float): MSE loss
    """
    loss = calculate_loss(y, tx, w, lambda_)
    grad = calculate_gradient(y, tx, w, lambda_)
    w = w - (grad * gamma).T
    return w, loss

# logistic regression without regularization

def logistic_regression_gradient_descent(y, tx, initial_w=0, max_iters=1000, gamma=0.01, threshold=1e-8, log=False, store=False):
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
        w (numpy.ndarray): ndarray with shape (n_iters,m) if store, else shape(m,1) 
        loss (Union[numpy.ndarray, float]): ndarray with shape (n_iters,) if store, else float
    """
    # init parameters
    loss = np.inf
    last_loss = np.inf
    losses = []
    ws = []
    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), tx]
    if type(initial_w) == np.ndarray:
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
            break
    # visualization
    if store:
        return ws, losses
    else:
        return w, loss


# Logistic regression with regularization

def reg_logistic_regression(y, tx, initial_w=0, max_iters=1000, gamma=0.01, threshold=1e-8, log=False, store=False, lambda_ = 0.1):
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
        w (numpy.ndarray): ndarray with shape (n_iters,m) if store, else shape(m,1) 
        loss (Union[numpy.ndarray, float]): ndarray with shape (n_iters,) if store, else float
    """
    # init parameters
    loss = np.inf
    last_loss = np.inf
    losses = []
    ws = []
    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), tx]
    if type(initial_w) == np.ndarray:
        ws = [initial_w]
        w = initial_w
    else:
        w = np.array([initial_w]*tx.shape[1])
        ws = [w]
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        last_loss = loss
        w, loss = learning_by_gradient_descent(y, tx, w, gamma,lambda_)
        # log info
        if log and iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        if store:
            losses.append(loss)
            ws.append(w)
        # converge criterion
        if np.abs(last_loss-loss) < threshold:
            break
    # visualization
    if store:
        return ws, losses
    else:
        return w, loss
