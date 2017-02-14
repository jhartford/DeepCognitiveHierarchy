#import train.train as train
from train import train
from data import GameData
import os.path
#import bogota.data
import argparse
import time
import json
#import mobdata

def kfold(fold_function, start_fold=0, end_fold=10):
    for i in xrange(start_fold, end_fold):
        print 'STARTING FOLD: %d' % (i + 1)
        t = time.time()
        fold_function(i)
        t = time.time() - t
        print 'Fold %d complete in %f seconds' % (i + 1 , t)
        print '*' * 100


def parse_args():
    parser = argparse.ArgumentParser(description="K-fold cross validation")
    parser.add_argument('--start_fold', default=0, type=int)
    parser.add_argument('--end_fold', default=10, type=int)
    #parser.add_argument('--resume', dest='resume', action='store_true')
    #parser.set_default(resume=False)
    parser.add_argument('--path', default='')
    parser.add_argument('--json', default=None,
        help="Path of json file describing the options of the experiment")
    return parser.parse_args()



def build_fold_function(options, new_experiment=False, resume=False):
    if not os.path.exists("./test/best_loss"):
        os.makedirs("./test/best_loss")
    output_file = options.get('path', './') + options.get('name', 'test') + '.csv'
    best_loss_filename = options.get('path', './') + "best_loss/" + options.get('name', 'test') + '.csv'
    par_file = options.get('path', './') + options.get('name', 'test') + '_%d_par.json'
    dataset_name = options.get('dataset', 'all9')
    seed = options.get('seed', 123)
    if not os.path.isfile(output_file):
        with open(output_file, 'w') as f:
            f.write('Data: %s, seed: %d\n' % (dataset_name, seed))
            f.write(','.join(['fold', 'seed', 'train', 'valid', 'test']) + '\n')

    def fold_function(k):  # k is the fold index
        data = GameData('./all9.csv', 50.)
        train_data, test_data = data.train_test(k, seed=seed)

        options['fold'] = k
        llk, best_par = train(options, [train_data.datalist(), test_data.datalist()], False)

        print "LLK: ", llk

        for kk, vv in best_par.items():
            temp = vv.tolist()
            del best_par[kk]
            best_par[kk] = temp
        log_fold(best_loss_filename, llk, k, options.get('model_seed', -99))

        with open(par_file % k, 'w') as f:
            json.dump(best_par, f)

    return fold_function


def log_fold(log_file_name, llk, fold, model_seed, llk_start=None):
    log_file_name = log_file_name.replace(".csv", "_out.csv")
    with open(log_file_name, 'a') as f:
        if llk_start is not None:
            log = [fold] + [model_seed] + list(llk) + list(llk_start)
        else:
            log = [fold] + [model_seed] + list(llk)
        f.write(','.join(str_lst(log)) + '\n')


def str_lst(x):
    return [str(i) for i in x]



DEFAULT_OPTIONS = {'name': 'test',
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
                   'max_itr': 3,
                   'model_seed': 3,
                   'objective': 'nll'}


def main():
    args = parse_args()
    if args.json is not None:
        options = json.load(open(args.json))
        print "OPTIONS FROM JSON LOADED SUCCESSFULLY."
        if args.path != '':
            options['path'] = args.path
    else:
        options = DEFAULT_OPTIONS
    fold_function = build_fold_function(options, args.start_fold==0, False)
    kfold(fold_function, args.start_fold, args.end_fold)


if __name__ == '__main__':
    main()



# def select_dataset(name):
#     if name=='all9':
#         return bogota.data.cn_all9
#     elif name=='all9_filtered':
#         return train.filter_games(bogota.data.cn_all9, 10)
#     elif name=='mob':
#         return mobdata.matrix_pool.matrix_pool
#     elif name=='all33':
#         return train.filter_games(bogota.data.cn_all9, [3, 3])
#     elif name=='all33_10':
#         return train.filter_games(bogota.data.cn_all10, [3, 3])
#     elif name=='all10':
#         return bogota.data.cn_all10
#     elif name=='all11':
#         return bogota.data.cn_all11
#     elif name=='combo9':
#         return bogota.data.cn_some9
#     elif name=='tiny9':
#         _, dat = bogota.data.cn_all9.train_fold_gamewise(101, 10, 0, True)
#         return dat
#     else:
#         raise NameError('Unknown dataset %s' % name)

# def build_fold_function(options, new_experiment=False, resume=False):
#     output_file = options.get('path', './') + options.get('name', 'test') + '.csv'
#     par_file = options.get('path', './') + options.get('name', 'test') + '_%d_par.json'
#     dataset_name = options.get('dataset', 'all9')
#     dataset = select_dataset(dataset_name)
#     seed = options.get('seed', 123)
#     if not os.path.isfile(output_file):
#         with open(output_file, 'w') as f:
#             f.write('Data: %s, seed: %d\n' % (dataset_name, seed))
#             f.write(','.join(['fold', 'seed', 'train', 'valid', 'test']) + '\n')
#
#     def fold_function(k):
#         data = train.get_data(dataset, k,
#                               normalise=options.get('normalise', 500.),
#                               seed=seed,
#                               validation=options.get('use_validation', True),
#                               strat=options.get('stratified', False))
#         options['fold'] = k
#         llk, best_par = train.train(options, data, resume)
#         for kk, vv in best_par.iteritems():
#             best_par[kk] = vv.tolist()
#             log_fold(output_file, llk, k, options.get('model_seed', -99))
#             with open(par_file % k, 'w') as f:
#                 json.dump(best_par, f, indent=2)
#
#     """
#         options = DEFAULT_OPTIONS
#     print 'Getting Data'
#     #data = get_data(bogota.data.cn_all9, 0, normalise=50., seed=101)
#     data = GameData('./all9.csv', 50.)
#     train_data, test_data = data.train_test(0, seed=101)
#     perf, par = train(options, [train_data.datalist(), test_data.datalist()], False)
#     """
#     return fold_function
