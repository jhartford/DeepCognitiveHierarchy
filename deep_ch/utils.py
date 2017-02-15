"""
Helper functions for game-net

Modified from:
https://github.com/ryankiros/visual-semantic-embedding/blob/master/utils.py
Which is mostly modified from:
https://github.com/fh295/DefGen2
"""

# Numerical libraries
import theano
import theano.tensor as tensor
import numpy

# Standard libraries
import warnings
from collections import OrderedDict


def init_rng(rng):
    if rng is None:
        print 'init rng'
        return numpy.random.RandomState()
    else:
        return rng


def zipp(params, tparams):
    """
    Push parameters to Theano shared variables
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

def unzip(zipped):
    """
    Pull parameters from Theano shared variables
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

def split_params(tparams, remove=[]):
    """
    Split an OrderedDict into two by specifying
    either what is to be removed
    """
    kept_params = OrderedDict()
    removed_params = OrderedDict()
    for kk, vv in tparams.iteritems():
        if kk in remove:
            removed_params[kk] = vv
        else:
            kept_params[kk] = vv

    return kept_params, removed_params

def itemlist(tparams):
    """
    Get the list of parameters.
    Note that tparams must be OrderedDict
    """
    return [vv for kk, vv in tparams.iteritems()]


def _p(pp, name):
    """
    Make prefix-appended name
    """
    return '%s_%s' % (pp, name)


def init_tparams(params):
    """
    Initialize Theano shared variables according to the initial parameters
    """
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def sync_tparams(params, tparams):
    """
    Initialize Theano shared variables if they don't already exist
    else set the shared variable to their value in params.
    """
    for kk, pp in params.iteritems():
        if kk in tparams:
            tparams[kk].set_value(pp)
        else:
            tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def load_params(path, params):
    """
    Load parameters
    """
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]
    return params


def floatx(x):
    return numpy.asarray(x, dtype=theano.config.floatX)


def ortho_weight(ndim, rng=None):
    """
    Orthogonal weight init, for recurrent layers
    """
    #rng = init_rng(rng)
    W = rng.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return floatx(u)


def norm_weight(nin, nout=None, scale=0.1, ortho=True, rng=None):
    """
    Uniform initalization from [-scale, scale]
    If matrix is square and ortho=True, use ortho instead
    """
    rng = init_rng(rng)
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin, rng)
    else:
        W = rng.uniform(low=-scale, high=scale, size=(nin, nout))
    return floatx(W)


def xavier_weight(nin, nout=None, rng=None):
    """
    Xavier init
    """
    rng = init_rng(rng)
    if nout is None:
        nout = nin
    r = numpy.sqrt(6.) / numpy.sqrt(nin + nout)
    W = rng.rand(nin, nout) * 2 * r - r
    return floatx(W)


def l2norm(X):
    """
    Compute L2 norm, row-wise
    """
    norm = tensor.sqrt(tensor.pow(X, 2).sum(1))
    X /= norm[:, None]
    return X


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


def par_to_lists(par):
    '''
    Convert numpy arrays to lists so that they can be written to json
    objects
    '''
    json_par = OrderedDict()
    for kk, vv in par.iteritems():
        json_par[kk] = vv.tolist()
    return json_par


def par_to_arrays(par):
    '''
    Convert lists to numpy arrays so that they can be read from json
    objects
    '''
    json_par = OrderedDict()
    for kk, vv in par.iteritems():
        json_par[kk] = numpy.array(vv)
    return json_par


def print_params(tparams):
    for kk, vv in tparams.iteritems():
        print kk
        if isinstance(vv, tensor.sharedvar.TensorSharedVariable):
            print vv.get_value()
        else:
            print vv
        print
