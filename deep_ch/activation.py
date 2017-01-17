import theano
import theano.tensor as tensor


def get_activ(name):
    '''
    str -> function

    Helper function for importing function by name
    '''
    return eval(name)


def softmax(x):
    '''
    Softmax activation function
    '''
    e_x = tensor.exp(x - x.max(axis=1, keepdims=True))
    out = e_x / e_x.sum(axis=1, keepdims=True)
    return out


def tanh(x):
    """
    Tanh activation function
    """
    return tensor.tanh(x)

def sig(x):
    """
    Linear activation function
    """
    return tensor.nnet.sigmoid(x)

def linear(x):
    """
    Linear activation function
    """
    return x


def relu(x):
    '''
    Rectified linear activation function
    '''
    return tensor.maximum(tensor.cast(0., theano.config.floatX), x)
