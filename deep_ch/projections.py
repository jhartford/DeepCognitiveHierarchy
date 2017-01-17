import simplex
import numpy

def get_operator(name):
	if name == 'simplex':
		return project_simplex
	elif name == 'bound':
		return bound
	else:
		raise NameError, 'Unkown operator %s' % name

def bound(x, a=0., b=1.):
	return numpy.clip(x, a, b)

def project_simplex(x):
    """
    Project an arbitary vector onto the simplex.
    See [Wang & Carreira-Perpin 2013] for a description and references.

    TODO: Implement in theano for faster projections
    """
    n = x.shape[0]
    mu = -numpy.sort(-x)  # sort decending
    sm = 0
    for j in xrange(1, n+1):
        sm += mu[j - 1]
        t = mu[j - 1] - (1./(j)) * (sm - 1)
        if t > 0:
            row = j
            sm_row = sm
    theta = (1. / row) * (sm_row - 1)
    return numpy.abs(numpy.maximum(x - theta, 0))