"""
Optimizers.

All take a theano symbolic scalar for the learning rate, the parameters,
gradients and the function inputs and cost function, and return a
theano function for updating a list of gradients
(with respect to each parameter), and updating the parameters.

Modified from:
https://github.com/ryankiros/visual-semantic-embedding/blob/master/utils.py
and https://github.com/fh295/DefGen2/blob/master/defgen.py
"""
import theano
import theano.tensor as tensor
import numpy
from utils import itemlist, floatx
from theano.tensor.shared_randomstreams import RandomStreams


# Helper function for getting the optimiser by name.
# Avoids having to import each optimiser individually
def get_optim(name):
    return eval(name)

def sgd(lr, tparams, grads, inp, cost, use_noise,**kwargs):
    print 'Using SGD'
    gshared = [theano.shared(p.get_value() * floatx(0.), name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, gs + g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, givens={use_noise: numpy.float32(1.)},
                                    on_unused_input='warn', updates=gsup, allow_input_downcast=True)

    pup = [(p, p - lr * (g))
           for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup, allow_input_downcast=True)

    return f_grad_shared, f_update, gshared


def adam(lr, tparams, grads, inp, cost, use_noise, **kwargs):
    '''
    See: Adam - a method for stochastic optimization. https://arxiv.org/abs/1412.6980

    Note that when using Adam, the lr learning rate parameter does nothing because Adam chooses 
    per-parameter learning rates. If you want to be able to manually turn down the learning rate,
    you can modify the parameter update line:
    p_t = p - (lr_t * g_t)
    to:
    p_t = p - lr * (lr_t * g_t)
    so that you have a global learning rate parameter.
    '''
    gshared = [theano.shared(p.get_value() * floatx(0.), name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, gs + g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup, allow_input_downcast=True)

    lr0 = floatx(0.0002)
    b1 = floatx(0.1)
    b2 = floatx(0.001)
    e = floatx(1e-8)

    updates = []

    i = theano.shared(floatx(0.))
    i_t = i + floatx(1.)
    fix1 = floatx(1.) - b1**(i_t)
    fix2 = floatx(1.) - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() *floatx( 0.))
        v = theano.shared(p.get_value() *floatx( 0.))
        m_t = (b1 * g) + ((floatx(1.) - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((floatx(1.) - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr], [], updates=updates,
                               on_unused_input='ignore', allow_input_downcast=True)

    return f_grad_shared, f_update, gshared

