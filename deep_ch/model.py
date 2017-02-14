'''
Module for initialising parameters and building the model
'''

import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy

from collections import OrderedDict

from utils import _p, ortho_weight, norm_weight, xavier_weight, l2norm
from layers import get_layer


def set_defaults(options):
    options['batch_size'] = options.get('batch_size', None)
    # Regularisation
    options['l1'] = options.get('l1', 0.0)
    options['l2'] = options.get('l2', 0.0)
    # Optimisation method
    options['opt'] = options.get('opt', 'adam')
    options['pooling'] = options.get('pooling', False)
    options['pooling_activ'] = options.get('pooling_activ', 'sum')
    options['lr'] = options.get('learning_rate', 1e-2)
    options['max_itr'] = options.get('max_itr', 10000)
    options['debug'] = options.get('debug', False)
    options['ar_layers'] = options.get('ar_layers', 1)
    options['dropout'] = options.get('dropout', False)
    options['dropout_rate'] = options.get('dropout_rate', 0.8)
    options['hidden_units'] = options.get('hidden_units', [0])
    options['shared_ld'] = options.get('shared_ld', True)
    options['activ'] = options.get('activ', ['linear'] *
                                   len(options['hidden_units']))
    # weighting of the validation set relative to the training set for
    # selecting best performing model
    options['percent_valid'] = options.get('percent_valid', 0.)
    options['model_seed'] = options.get('model_seed', None)
    return options


def init_params(options, rng=None):
    """
    Initialize all network parameters and constrains.
    
    All parameters and their corresponding constraints are stored in an OrderedDict.
    """
    params = OrderedDict()
    constraints = OrderedDict()

    input_size = 2 # number of player utilities

    n_hidden = [input_size] + options['hidden_units']
    for i in xrange(1, len(n_hidden)):
        params = get_layer('hid')[0](options, params, prefix='hidden%02d' % i,
                                     nin=n_hidden[i-1] * (3 if options['pooling'] else 1), # 3 x parameters if pooling used
                                     nout=n_hidden[i],
                                     rng=rng, b_offset=1. )
    params = get_layer('softmax')[0](options, params, nin=n_hidden[-1],
                                     rng=rng)

    ar_layers = options['ar_layers']

    for i in range(ar_layers):
        for p in range(2):
            if i == ar_layers - 1 and p == 1:
                # don't build ar layer for pl 2 in the last layer because it is not used
                continue
            params, constraints = get_layer('ar')[0](options, params,
                                        prefix='p%d_ar%d' % (p, i),
                                        nin=ar_layers, level=i, rng=rng,
                                        constraints=constraints)
    params, constraints = get_layer('output')[0](options, params, constraints, rng=rng, nin=ar_layers)
    return params, constraints

def to_list(x, n):
    return x if isinstance(x, list) else [x] * n

def build_features(x, tparams, options, use_noise, trng, normalise=True):
    '''
    Build the Feature Layers components of the network.
    '''
    use_dropout = options['dropout']
    if use_dropout:
        print 'Using dropout'
    prev = x
    
    hidden_outputs = []
    n_hidden = len(options['hidden_units'])
    activ = to_list(options['activ'], n_hidden)

    for i in xrange(n_hidden):
        if options['pooling']:
            # Add pooling units
            prev = get_layer('pooling')[1](tparams, prev, options, activ=options['pooling_activ'])
        prev = get_layer('hid')[1](tparams, prev, options,
                                   prefix='hidden%02d' % (i + 1),
                                   activ=activ[i])
        hidden_outputs.append(prev)
        if use_dropout:
            prev = get_layer('dropout')[1](prev, use_noise, options, trng)

    out = get_layer('sum')[1](tparams, prev, options)
    
    if normalise:
        out = get_layer('softmax')[1](tparams, out, options)
    return out, hidden_outputs


def build_ar_layers(x, tparams, options, features, hiddens):
    u1, u2 = (x[:, 0, :, :], x[:, 1, :, :].transpose(0, 2, 1))

    h1, h2 = hiddens
    # concatinate the payoff matrix onto the final layer hidden units
    utility = (tensor.concatenate((u1.reshape((u1.shape[0],
                                                1,
                                                u1.shape[1],
                                                u1.shape[2])), h1), axis=1),
                tensor.concatenate((u2.reshape((u2.shape[0],
                                                1,
                                                u2.shape[1],
                                                u2.shape[2])), h2), axis=1))

    ar_layers = options['ar_layers']

    ar_lists = ([], [])
    opp = [None, None]
    weighted_feature_list = ([], [])
    br_list = ([], [])
    for i in range(ar_layers):
        for p in range(2):
            if i == (ar_layers - 1) and p == 1:
                continue  # don't build ar layer for pl 2 in the last layer
            feat = features[p]
            ar, weighted_features, br = get_layer('ar')[1](tparams,
                                                           feat,
                                                           options,
                                                           payoff=utility[p],
                                                           prefix='p%d_ar%d' % (p, i),
                                                           opposition=opp[p],
                                                           level=i)
            n, d = ar.shape
            ar = ar.reshape((n, 1, d))  # make space to concat ar layers
            weighted_feature_list[p].append(weighted_features)
            if i == 0:
                ar_lists[p].append(ar)
            else:
                ar_lists[p].append(tensor.concatenate((ar_lists[p][i - 1], ar),
                                   axis=1))
                br_list[p].append(br)

        # append each layer then update the opposition variable...
        if i < ar_layers - 1:
            for p in range(2):
                opp[1 - p] = ar_lists[p][i]
    # return ar_lists[0][ar_layers-1]
    return ar_lists, weighted_feature_list, br_list


def build_model(tparams, options, rng=None):
    """
    Computation graph for the model
    """
    if rng is None:
        rng = numpy.random.RandomState(123)
    trng = RandomStreams(rng.randint(1000000))
    use_noise = theano.shared(numpy.float32(0.))
    x = tensor.tensor4('x')

    own_features, hidden1 = build_features(x, tparams, options, use_noise, trng)
    opp_features, hidden2 = build_features(x.transpose((0, 1, 3, 2))[:, [1, 0], :, :], # transpose to get player 2 model
                                           tparams, options, use_noise, trng)

    ar, weighted_feature_list, br_list = build_ar_layers(x, tparams, options,
                                                         (own_features,
                                                          opp_features),
                                                         (hidden1[-1],
                                                          hidden2[-1]))
    ar_layers = options['ar_layers']
    out = get_layer('output')[1](tparams, ar[0][ar_layers-1], options)

    intermediate_fns = {'ar': ar,
                        'own_features': own_features,
                        'opp_features': opp_features,
                        'hidden1': hidden1,
                        'hidden2': hidden2,
                        'weighted_feature_list': weighted_feature_list,
                        'br_list': br_list}
    if not options['debug']:
        return trng, use_noise, x, out
    else:
        return trng, use_noise, x, out, intermediate_fns
