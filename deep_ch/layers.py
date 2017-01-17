'''
Layers for building model


Built on the following code:
https://github.com/ryankiros/visual-semantic-embedding/blob/master/utils.py
... which is mostly modified from:
https://github.com/fh295/DefGen2
'''
import theano
import theano.tensor as tensor

import numpy

from activation import get_activ
from utils import _p, floatx, ortho_weight, norm_weight, xavier_weight, \
    sync_tparams, init_rng

# clip bound
bound = 1e16

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'hid': ('param_init_hid_layer', 'hid_layer'),
          'pooling': ('param_init_pooling', 'pooling'),
          'max': ('param_init_max_layer', 'max_layer'),
          'mean': ('param_init_mean_layer', 'mean_layer'),
          'sum': ('param_init_max_layer', 'sum_layer'),
          'softmax': ('param_init_softmax_layer', 'softmax_layer'),
          'dropout': ('None', 'dropout_layer'),
          'ar': ('param_init_action_response_layer', 'action_response_layer'),
          'output': ('param_init_output', 'output_layer')
          }


def get_layer(name):
    """
    get_layer: str -> function

    Return param init and feedforward functions for the given layer name
    """
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# Hidden layer
def param_init_hid_layer(options, params, prefix='hidden',
                         nin=None, nout=None, rng=None, init=xavier_weight, b_offset=0.):
    if nin is None:
        nin = options['n_players']
    params[_p(prefix, 'W')] = init(nin, nout, rng=rng)
    params[_p(prefix, 'b')] = floatx(b_offset + numpy.zeros((nout,)))
    return params


def hid_layer(tparams, state_below, options,
              prefix='hidden', activ='linear', **kwargs):
    """
    hid_layer: tensor4 -> tensor4
    (iter, player, action1, action2) -> (iter, feature, action1, feat_payoff)

    Feedforward pass on tensors rather than cells in matrices.

    used to calculate \sum_{i = 1}^n w_i X_i over a minibatch.

    We flatten the input tensor down to an (n_inputs, batchsize*n*d)
    and then multiply by the (n_inputs, n_units matrix) using
    broadcasting. This creates an (n_inputs, n_units, batchsize*n*d)
    tensor which is then sumed over the n_inputs axis before reshaping
    back to a (n_units, batchsize, n, d) tensor...
    """
    n, p, i, j = state_below.shape
    w = tparams[_p(prefix, 'W')]
    nin, nout = w.shape
    b = tparams[_p(prefix, 'b')]

    # weighted sum over players (second axis of tensor)
    ws = w.dimshuffle(0, 1, 'x') *\
        state_below.transpose((1, 0, 2, 3)).\
        reshape((nin, n*i*j)).dimshuffle(0, 'x', 1)
    ws = ws.sum(axis=0)
    # add bias
    ws += b.dimshuffle(0, 'x')
    ws = ws.reshape((nout, n, i, j))
    ws = ws.transpose(1, 0, 2, 3)

    # Apply the nonlinearity
    return get_activ(activ)(ws)


def param_init_pooling(options, params, prefix='pooling',rng=None, **kwargs):
    '''
    No parameters
    '''
    return params

def pooling(tparams, state_below, options, prefix='pooling', activ='sum', **kwargs):
    fn = eval('tensor.%s' % activ)
    x = state_below
    # row-wise pooling
    rw = fn(x, axis=2)
    o_r = tensor.ones_like(x[0, 0, :, 0])
    # reshape back into a matrix
    rw = (rw * o_r[:, None, None, None]).transpose((1,2,0,3))
    # column-wise pooling
    cw = fn(x, axis=3)
    o_c = tensor.ones_like(x[0, 0, 0, :])
    # reshape back into a matrix
    cw = (cw * o_c[:, None, None, None]).transpose((1,2,3,0))

    out = tensor.concatenate((x, rw), axis=1)
    out = tensor.concatenate((out, cw), axis=1)
    return out


# Pooling layers for reducing the final hidden units into vectors
# In the NIPS paper we use the sum layer to sum uniformly over actions,
# but mean and max could also be tested.

def param_init_max_layer(options, params, **kwargs):
    '''
    Max layer doesn't have any parameters... initialization here for
    consistency.
    '''
    return params


def max_layer(tparams, state_below, options,
              prefix='max', **kwargs):
    """
    max_layer: tensor4 -> tensor3
    (iter, feature, action1, feat_payoff) -> (iter, feature, action_payoff)

    Rowise max of the the input tensor
    """
    axis = state_below.ndim - 1
    return tensor.max(state_below, axis=axis)

def param_init_mean_layer(options, params, **kwargs):
    '''
    Mean layer doesn't have any parameters... initialization here for
    consistency.
    '''
    return params

def mean_layer(tparams, state_below, options,
              prefix='mean', **kwargs):
    """
    max_layer: tensor4 -> tensor3
    (iter, feature, action1, feat_payoff) -> (iter, feature, action_payoff)

    Rowise sum of the the input tensor
    """
    axis = state_below.ndim - 1
    return tensor.mean(state_below, axis=axis)


def sum_layer(tparams, state_below, options,
              prefix='sum', **kwargs):
    """
    max_layer: tensor4 -> tensor3
    (iter, feature, action1, feat_payoff) -> (iter, feature, action_payoff)

    Rowise sum of the the input tensor
    """
    axis = state_below.ndim - 1
    return tensor.sum(state_below, axis=axis)

def param_init_softmax_layer(options, params, prefix='softmax', **kwargs):
    '''
    None
    '''
    return params


def softmax_layer(tparams, state_below, options,
                  prefix='softmax', **kwargs):
    """
    softmax_layer: tensor3 -> tensor3
    (iter, feature, action_payoff) -> (iter, feature, prob of action)

    Softmax caluclates the probability of an action given its payoff.
    """
    n, f, i = state_below.shape
    out = tensor.nnet.softmax(state_below.reshape((n*f, i)))
    return out.reshape((n, f, i))


def init_level_dist(params, upper_bound, nin, rng, constraints):
    '''
    Initialize a shared level distribution in the action response layers. 
    '''
    if 'ld' not in params:
        params['ld'] = floatx(rng.uniform(size=(nin),
                                          low=0.1,
                                          high=0.9))
        # constrain level distribution to be on the simplex
        params['ld'] /= params['ld'].sum()
        constraints['simplex'] = constraints.get('simplex', []) + ['ld']
    return params, constraints


def param_init_action_response_layer(options, params, constraints, prefix='ar',
                                     nin=0, rng=None, unif_range=0.2,
                                     level=0, **kwargs):
    '''
    Action response layers.
    '''
    rng = init_rng(rng)
    n_features = options['hidden'][-1]

    if options['shared_ld']:
        params, constraints = init_level_dist(params, unif_range, nin, rng, constraints,
                                            simplex=simplex)
    else:
        if level > 0:
            params[_p(prefix, 'ld')] = floatx(rng.uniform(size=(level),
                                              low=0.1,
                                              high=0.9))
            params[_p(prefix, 'ld')] /= params[_p(prefix, 'ld')].sum()
            constraints['simplex'] = constraints.get('simplex', []) + [_p(prefix, 'ld')]

    initial_Wf = numpy.zeros(n_features)
    initial_Wf += floatx(rng.uniform(size=(n_features), low=0., high=unif_range))
    initial_Wf /= initial_Wf.sum()

    if level == 0:
        params[_p(prefix, 'Wf')] = floatx(initial_Wf)
        constraints['simplex'] = constraints.get('simplex', []) + [_p(prefix, 'Wf')]

    if level > 0:
        params[_p(prefix, 'W_h')] = floatx(rng.uniform(size=(1+options['hidden'][-1]),
                                            low=-0.01,
                                            high=0.01))
    if level > 0:
        params[_p(prefix, 'lam')] = floatx(1.0)
    return params, constraints


def action_response_layer(tparams, features, options, payoff=None,
                          prefix='ar', opposition=None, level=0, **kwargs):
    """
    action_response_layer:  tensor3, (tensor3) -> matrix
                            features, (opposition) -> ar_layer

    Tensor dims:
    features: iter, action_payoff, feature
    opposition: iter, level, prob of action
    output: iter, prob of action

    Probability of an action given features and beliefs about opposition.
    """
    n, f, i = features.shape

    # Weights on opposition players
    if level == 0:
        w_feat = tparams[_p(prefix, 'Wf')]
        weighted_features = tensor.sum(features * w_feat.dimshuffle('x', 0, 'x'), axis=1)

        ar = weighted_features
        return ar, weighted_features, None
    else:
        weighted_features = None
        lam = tparams[_p(prefix, 'lam')] 
        if options['shared_ld']:
            level_dist = tparams['ld']
            ld = level_dist
            ld += floatx(1e-32) # avoid divide by zero
            ld = ld[0:level]
            ld /= ld.sum()
        else:
            ld = tparams[_p(prefix, 'ld')]
            ld += floatx(1e-32)
            ld /= ld.sum()
        
        # U * AR * ld (where * is matrix product)
        weighting = opposition * ld.dimshuffle('x', 0, 'x')
        prob_a = tensor.sum(weighting, axis=1)


        payoff = payoff * tparams[_p(prefix, 'W_h')].dimshuffle('x', 0, 'x', 'x')
        payoff = tensor.sum(payoff,axis=1)
        
        br = tensor.sum(payoff * prob_a.dimshuffle(0, 'x', 1), axis=2)
        out = br
        # remove weighted_features, br when done with visualisation
        return tensor.nnet.softmax(out * lam), weighted_features, br

def param_init_output(options, params, constraints, rng, nin, prefix=''):
    if options['shared_ld']:
        return params, constraints
    else:
        params['ld'] = floatx(rng.uniform(size=(nin),
                                          low=0.1,
                                          high=0.9))
        # constrain level distribution to be on the simplex
        params['ld'] /= params['ld'].sum()
        constraints['simplex'] = constraints.get('simplex', []) + ['ld']
        return params, constraints


def output_layer(tparams, ar_layers, options,
                 prefix='out', **kwargs):
    """
    action_response_layer:  tensor3 -> matrix
                            action_response_layers -> prob_a

    Tensor dims:
    ar_layer: iter, ar_layer, action
    output: iter, prob of action

    Probability of an action given features and beliefs about opposition.
    """
    # Weights on features

    ld = tparams['ld']
    out = tensor.sum(ar_layers * ld.dimshuffle('x', 0, 'x'), axis=1)
    return out
