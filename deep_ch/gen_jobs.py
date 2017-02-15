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

def main():
    settings = {}
    if len(sys.argv) == 2:
        settings = defaults
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
