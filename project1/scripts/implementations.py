import numpy as np
import numpy.linalg as lin

#Base Functions
def compute_MSE_loss(y, tx, w):
    """Calculate the MSE loss.
        Parameters :
            y (numpy.ndarray): An array with shape (n,1)
            tx (numpy.ndarray): An array with shape (n,m)
            w (numpy.ndarray): An array with shape (k,1)
        Returns:
            loss (numpy.float64): the MSE loss
    """
    return (1/(2*len(y)))* np.sum((y-tx.dot(w))**2)
    

def compute_gradient(y, tx, w):
    """Compute the gradient.
            Parameters:
                    y (numpy.ndarray): An array with shape (n,1)
                    tx (numpy.ndarray): An array with shape (n,m)
                    w (numpy.ndarray): An array with shape (m,1) describing the model
            Returns:
                    grad (numpy.ndarray): An array with shape (m,1), the gradient
    """

    ew=y-tx.dot(w.T)
    grad= -(1/len(y))*(tx.T.dot(ew))
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
    ew=y-tx.dot(w.T)
    grad= -(1/len(y))*(tx.T.dot(ew))
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


# least squares gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma, log=False):
    """Compute the Gradiant Descent with least squares loss function.
            Parameters:
                    y (numpy.ndarray): An array with shape (n,1)
                    tx (numpy.ndarray): An array with shape (n,m)
                    initial_w (numpy.ndarray): An array with shape (m,1) describing the model
                    max_iters (int) : number of iterations
                    gamma (numpy.float64): gradiant multiplier gamma > 0 
                    log (bool): if set to True display step by step informations about the descent
            Returns:
                    losses (numpy.ndarray): An array with shape (max_iter,1) containing the square loss of each iteration
                    ws (numpy.ndarray): An array with shape (max_iter,m) containing the w for each iteration 
    """
    # Define parameters to store w and loss
    
    ws = [initial_w]

    losses = []
    w = initial_w
    for n_iter in range(max_iters): 
        # compute gradient, loss and next iter w
        grad=compute_gradient(y,tx,w)
        loss=compute_MSE_loss(y,tx,w)
        w=(-gamma*grad).T+w
        # store w and loss
        ws.append(w)
        losses.append(loss)
        if(log):
          print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

# stochastic gradient descent

def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, batch_size=1, log=False):
    """Compute the Stochastic Gradiant Descent with least squares loss function.
            Parameters:
                    y (numpy.ndarray): An array with shape (n,1)
                    tx (numpy.ndarray): An array with shape (n,m)
                    initial_w (numpy.ndarray): An array with shape (m,1) describing the model
                    max_iters (int) : number of iterations
                    gamma (numpy.float64): gradiant multiplier gamma > 0 
                    log (bool): if set to True display step by step informations about the descent
                    batch_size (int): default to 1, number of data_points used to compute the gradient  
                    log (bool): if set to True display step by step informations about the descent
            Returns:
                    losses (numpy.ndarray): An array with shape (max_iter,1) containing the square loss of each iteration
                    ws (numpy.ndarray): An array with shape (max_iter,m) containing the w for each iteration 
    """
    iterator=batch_iter(y, tx, batch_size, num_batches=max_iters, shuffle=True)
    ws = [initial_w]
    losses = []
    w = initial_w
    for y,x in iterator:
        # compute gradient and loss
        grad=compute_stoch_gradient(y,x,w)
        loss=compute_MSE_loss(y,x,w)
        # update w by gradient
        w=(-gamma*grad).transpose()+w
        # store w and loss
        ws.append(w)
        losses.append(loss)
        if(log):
          print("Gradient Descent(): loss={l}, w0={w0}, w1={w1}".format(
              l=loss, w0=w[0], w1=w[1]))

    return losses, ws

# least squares
def least_squares(y, tx):
    """calculate the least squares solution.
            Parameters:
                    y (numpy.ndarray): An array with shape (n,1)
                    tx (numpy.ndarray): An array with shape (n,m)
            Returns:
                    w* (numpy.ndarray): An array with shape (m,1) the optimal model of complexity k
    """
    w= lin.inv(tx.T.dot(tx)).dot(tx.T).dot(y)
    return ((1/(2*len(y)))* np.sum((y-tx.dot(w))**2),w)

# ridge regression
def ridge_regression(y, tx, lambda_):
    """implement ridge regression.
            Parameters:
                    y (numpy.ndarray): An array with shape (n,1)
                    tx (numpy.ndarray): An array with shape (n,m)
                    lambda_ (numpy.float64): regularization factor lambda_>0
            Returns:
                    w* (numpy.ndarray): An array with shape (m,1) the optimal model of complexity k
    """
    lambda_p=2*len(y)*lambda_
    w=lin.inv(tx.T.dot(tx)+lambda_p*np.identity(len(tx[0]))).dot(tx.T).dot(y)
    return w