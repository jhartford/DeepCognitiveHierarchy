import json
import sys
import os
from collections import OrderedDict

defaults = {'name': 'test',
                   'save_path': './',
                   'hidden_units': [50, 50],
                   'activ': 'relu',
                   'pooling': True,
                   'batch_size': None,
                   'ar_layers': 1,
                   'dropout': False,
                   'l1': 0.01,
                   'l2': 0.0,
                   'pooling_activ':'max',
                   'opt': 'adam',
                   'max_itr': 25000,
                   'model_seed': 3,
                   'objective': 'nll'}

keys = ['name', 'save_path', 'hidden_units', 'activ', 'pooling', 'batch_size', 'ar_layers', 'dropout',
        'l1', 'l2', 'pooling_activ', 'opt', 'max_itr', 'model_seed', 'objective']

# defaults = {'name': 'dirtest3', 'dataset': 'all9', 'normalise': 500.,
#                 'seed':187, 'hidden': [2], 'activ': ['relu'],
#                                 'col_hidden': [5],
#                 'n_layers': 7, 'dropout': False, 'l1': 0.0, 'learning_rate': 1e-1,
#                 'min_lr': 1e-10, 'max_lr': 10.,
#                 'opt': 'sgd', 'backtracking':True, 'percent_valid':0., 'simplex': True, 'ar_sharpen': 100.,
#                 'max_itr': 10000, 'lam_scaling':50., 'anneal': False, 'model_seed': 123}
# keys = ['name', 'dataset', 'normalise','seed', 'hidden', 'col_hidden', 'activ', 'n_layers', 'dropout', 'l1', 'learning_rate',
#         'min_lr', 'max_lr', 'opt', 'backtracking', 'simplex', 'max_itr', 'lam_scaling', 'ar_sharpen', 'anneal', 'percent_valid', 'model_seed']

#PATH = '001_Oct'


def main():
    settings = OrderedDict()       # Why is settings an ordered dict?
    if len(sys.argv) == 2:
        for k in keys:
            settings[k] = defaults[k]
    else:
        settings = json.load(open(sys.argv[2]))
    print "SETTINGS: ", settings
    settings['path'] = os.path.abspath(sys.argv[1]) + '/'
    name = sys.argv[1][0:-1]

    for seed in range(101, 111):
        settings['seed'] = seed
        for i in range(1):
            filename = '%s%s_%03d_%02d' % (settings['path'], name, seed, i)
            settings['model_seed'] = i
            settings['name'] = '%s_%d_%02d' % (name, seed, i)
            print(filename)
            with open(filename+'.json', 'w') as f:
                json.dump(settings, f, indent=2)


# Usage: argv[1] should be a path to a writable directory where json files would go
if __name__ == '__main__':
    main()
