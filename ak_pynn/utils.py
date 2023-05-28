import numpy as np
from opt_einsum import contract

# Activation functions
def sigmoid( z, derive=False):
    '''
    Performs the Sigmoid activation or Sigmoid gradient on a given set of inputs
    Note: Works for 2D arrays only(rows for samples, columns for nodes/outputs)
    
    Parameters:
    z:  (N,k) ndarray (N: no. of samples, k: no. of nodes)
    derive: boolean value (True - gradients,False - activations ) | Default- False
    
    Returns:
    ndarray : Sigmoid activated (N,k) if derive=False | Sigmoid gradient (N,k) if derive=True
    '''
    
    '''
    This Clipping is necessary before calculating exponents
    , because exp(large_number) will produce overflow error
    so clipping inputs with max range of 700
    will help to handle this error.
    
    >>np.exp(700) is almost close to inf but not inf
    >>np.exp([>700]) will produce overflow error
    '''
    
    z=np.clip(z,a_min=-700,a_max=700)
    if (derive):
        z=sigmoid(z)
        return (z*(1-z))
    else:
        return (1/(1+np.exp(-z)))


def softmax(z,derive=False):
    '''
    Performs the Softmax activation or Softmax gradient on a given set of inputs
    Note: Works for 2D arrays only(rows for samples, columns for nodes/outputs)
    
    Parameters:
    z:  (N,k) ndarray (N: no. of samples, k: no. of nodes)
    derive: boolean value (True - gradients,False - activations ) | Default- False
    
    Returns:
    ndarray : Softmax activated (N,k) if derive=False | Softmax gradient (N,k) if derive=True
    '''
    
    if (derive):
        z=softmax(z)
        temp = np.zeros((z.shape[0], z.shape[1], z.shape[1]))
        for i in range(z.shape[0]):
            sz=z[i].reshape(-1,1)
            temp[i]=np.diagflat(sz)-np.dot(sz, sz.T)
        return temp
    else:
        e_x = np.exp(z - np.max(z,axis=1).reshape(-1,1))
        return e_x / e_x.sum(axis=1).reshape((-1, 1))

def softmaxTimesVector(a,b): 
    output = contract('ik,ijk->ij',a,b, dtype=a.dtype)
    return output

def relu(z, derive=False):
    '''
    Performs the ReLU activation or ReLU gradient on a given set of inputs
    Note: Works for 2D arrays only(rows for samples, columns for nodes/outputs)
    
    Parameters:
    z:  (N,k) ndarray (N: no. of samples, k: no. of nodes)
    derive: boolean value (True - gradients,False - activations ) | Default- False
    
    Returns:
    ndarray : ReLU activated (N,k) if derive=False | ReLU gradient (N,k) if derive=True
    '''
    
    if (derive):
        return np.where(z >= 0, 1, 0)
    else:
        return np.where(z >= 0, z, 0)

def tanh(z, derive=False):

    '''
    Performs the Tanh activation or Tanh gradient on a given set of inputs
    Note: Works for 2D arrays only(rows for samples, columns for nodes/outputs)
    
    Parameters:
    z:  (N,k) ndarray (N: no. of samples, k: no. of nodes)
    derive: boolean value (True - gradients,False - activations ) | Default- False
    
    Returns:
    ndarray : Tanh activated (N,k) if derive=False | Tanh gradient (N,k) if derive=True
    '''
    
    if (derive):
        return (1-np.tanh(z)**2)
    else:
        return np.tanh(z)

def elu(z, derive=False,elu_alpha=0.1):
    
    '''
    Performs the Elu activation or Elu gradient on a given set of inputs
    Note: Works for 2D arrays only(rows for samples, columns for nodes/outputs)
    
    Parameters:
    z:  (N,k) ndarray (N: no. of samples, k: no. of nodes)
    derive: boolean value (True - gradients,False - activations ) | Default- False
    
    Returns:
    ndarray : Elu activated (N,k) if derive=False | Elu gradient (N,k) if derive=True
    '''

    z=np.clip(z,a_min=-700,a_max=700)
    if (derive):
        return np.where(z >= 0, 1, elu(z)+elu_alpha)
    else:
        return np.where(z >= 0, z, elu_alpha*(np.exp(z)-1 ))

def leaky_relu(z, derive=False,leaky_relu_fraction=0.01):

    '''
    Performs the Leaky ReLU activation or Leaky ReLU gradient on a given set of inputs
    Note: Works for 2D arrays only(rows for samples, columns for nodes/outputs)
    
    Parameters:
    z:  (N,k) ndarray (N: no. of samples, k: no. of nodes)
    derive: boolean value (True - gradients,False - activations ) | Default- False
    
    Returns:
    ndarray : Leaky ReLU activated (N,k) if derive=False | Leaky ReLU gradient (N,k) if derive=True
    '''
    if (derive):
        return np.where(z >= 0, 1,leaky_relu_fraction)
    else:
        return np.where(z >= 0, z,leaky_relu_fraction*z)
        
def linear(z, derive=False):
    
    '''
    Performs the Linear activation or Linear gradient on a given set of inputs
    Note: Works for 2D arrays only(rows for samples, columns for nodes/outputs)
    
    Parameters:
    z:  (N,k) ndarray (N: no. of samples, k: no. of nodes)
    derive: boolean value (True - gradients,False - activations ) | Default- False
    
    Returns:
    ndarray : Linear activated (N,k) if derive=False | Linear gradient (N,k) if derive=True
    '''
    if (derive):
        return np.ones(z.shape)
    else:
        return z


def softplus(z,derive=False): 
    '''
    Performs the softplus activation or softplus gradient on a given set of inputs
    Note: Works for 2D arrays only(rows for samples, columns for nodes/outputs)
    
    Parameters:
    z:  (N,k) ndarray (N: no. of samples, k: no. of nodes)
    derive: boolean value (True - gradients,False - activations ) | Default- False
    
    Returns:
    ndarray : softplus activated (N,k) if derive=False | softplus gradient (N,k) if derive=True
    '''
    if derive is True:
        return sigmoid(z)
    else:
        return np.log(np.exp(z)+1)

# Loss functions
def mae(y,p):
    """
    Computes Mean Absolute error/loss between targets
    and predictions. 
    Input: p- predictions (N, k) ndarray (N: no. of samples, k: no. of output nodes)
           y- targets (N, k) ndarray (N: no. of samples, k: no. of output nodes)
    Returns: scalar

    """
    return np.mean(np.mean(np.abs(y - p),axis=0))


def mse(y, p):
    """
    Computes Mean Squared error/loss between targets
    and predictions. 
    Input: p- predictions (N, k) ndarray (N: no. of samples, k: no. of output nodes)
           y- targets (N, k) ndarray (N: no. of samples, k: no. of output nodes)
    Returns: scalar

    """
    return np.mean(np.mean((y - p) ** 2,axis=0))

def binary_cross_entropy(y, p):
    """
    Computes binary cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: p- predictions (N, k) ndarray (N: no. of samples, k: no. of output nodes)
           y- targets (N, k) ndarray        (N: no. of samples, k: no. of output nodes)
    Returns: scalar

    """
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.mean(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p)),axis=0))

def categorical_cross_entropy(y, p):
    """
    Computes categorical cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: p- predictions (N, k) ndarray
           y- targets (N, k) ndarray        
    Returns: scalar
    """
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.mean(-np.sum(y*np.log(p),axis=1))
   
def MAE_grad(y,p):
    """
    Computes mean Absolute error gradient between targets
    and predictions. 
    Input: p- predictions (N, k) ndarray (N: no. of samples, k: no. of output nodes)
           y- targets (N, k) ndarray (N: no. of samples, k: no. of output nodes)
    Returns: (N,k) ndarray

    """
    return -np.sign(y-p)


def MSE_grad(y,p):
    """
    Computes mean squared error gradient between targets
    and predictions. 
    Input: p- predictions (N, k) ndarray (N: no. of samples, k: no. of output nodes)
           y- targets (N, k) ndarray (N: no. of samples, k: no. of output nodes)
    Returns: (N,k) ndarray

    """
    return (-2*(y-p))

def BCE_grad(y,p):
    """
    Computes binary cross entropy gradient between targets (encoded as one-hot vectors)
    and predictions. 
    Input: p- predictions (N, k) ndarray (N: no. of samples, k: no. of output nodes)
           y- targets (N, k) ndarray        (N: no. of samples, k: no. of output nodes)
    Returns: (N,k) ndarray

    """
    epsilon=1e-07
    p += epsilon
    return (-(y/p)+((1-y)/(1-p)))
    
def CCE_grad(y,p):

    """
    Computes categorical cross entropy gradient between targets (encoded as one-hot vectors)
    and predictions. 
    Input: p- predictions (N, k) ndarray
           y- targets (N, k) ndarray        
    Returns: matrix
    """
    epsilon=1e-07
    p += epsilon
    return -(y/p)
