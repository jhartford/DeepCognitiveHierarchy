"""
Convert bogota datapool into a numpy matrix of examples.
"""
from operator import add, mul
import numpy as np

try:
    import bogota.datapool
    from bogota.utils import action_profiles
except:
    'Bogota import failed...'


class GameData(object):
    def __init__(self, filename=None, normalize=1.):
        self._data = {}
        if filename is not None:
            self.read_csv(filename, normalize)

    def add_game(self, payoffs, actioncounts, shape):
        if shape in self._data:
            self._data[shape][0] = np.vstack((self._data[shape][0], payoffs))
            self._data[shape][1] = np.vstack((self._data[shape][1], actioncounts))
        else:
            payoffs = payoffs.reshape((1, payoffs.shape[0]))
            actioncounts = actioncounts.reshape((1, actioncounts.shape[0]))
            self._data[shape] = [payoffs, actioncounts]

    def datalist(self):
        return self._data

    def read_csv(self, filename, normalize=1.):
        with open(filename) as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                else:
                    title, shape, payoffs, actioncounts = line.split('\t')
                    shape = tuple(eval(shape))
                    payoffs = np.array(eval(payoffs)) / float(normalize)
                    actioncounts = np.array(eval(actioncounts))
                    self.add_game(payoffs, actioncounts, shape)

    def _generate_game_dictionary(self):
        game_dict = {}
        for shape, games in self._data.iteritems():
            for payoff, actions in zip(games[0], games[1]):
                idx = tuple(payoff)
                if idx in game_dict:
                    game_dict[idx] = (shape, game_dict[idx][1] + actions)
                else:
                    game_dict[idx] = (shape, actions)
        return game_dict

    def _split_indices_into_folds(self, indices, num_folds):
        n = len(indices)
        s = n / num_folds    # number of samples in a fold
        fold_indices = [list(indices)[i * s:(i + 1) * s] for i in range(num_folds)]
        print len(fold_indices[0])
        print fold_indices
        for idx, i in enumerate(list(indices)[s * num_folds:]):
            fold_indices[idx] += [i]
            print "Number of samples in a fold: ", len(fold_indices[idx])
        print len(fold_indices[0])
        print fold_indices
        # Returns a list of folds where each fold contains indices of samples in X
        return fold_indices

    def train_test(self, fold, num_folds=10, seed=123):
        num_samples = 0
        for shape, games in self._data.items():
            num_samples += len(games[0])
        print "Number of samples: ", num_samples

        rng = np.random.RandomState(seed)
        gd = self._generate_game_dictionary()  # generated dict of {[payoffs]: ([shape], [actions])}
        game_idx = gd.keys()  # list of all payoffs
        indices = range(len(gd.keys()))  # range of the number of games in game_dict
        rng.shuffle(indices)
        fold_indices = self._split_indices_into_folds(indices, num_folds)
        train = GameData()
        test = GameData()
        for i, fold_idx in enumerate(fold_indices):
            for idx in fold_idx:
                shape, actioncount = gd[game_idx[idx]]
                if i != fold:
                    train.add_game(np.array(game_idx[idx]), actioncount, shape)
                else:
                    test.add_game(np.array(game_idx[idx]), actioncount, shape)
        return train, test


def write_datapool(pool, filename):
    '''
    Given a Bogota datapool, write it to file.
    '''
    with open(filename, 'w') as f:
        f.write('title\tshape\tpayoffs\tactioncounts\n')
        for wp in pool:
            shape = [len(pl.strategies) for pl in wp.game.players]
            dnp = np.array(wp.denormalized_profile())
            title = '.'.join(wp.game.title.split('.')[2:])
            player1_actions = sum([dnp[i] for i in range(shape[0])])
            player2_actions = wp.n - player1_actions
            player1payoffs = encode_game(wp.game, shape)
            Y = dnp[0:shape[0]]
            X = player1payoffs
            f.write('\t'.join([str(i) for i in [title, shape, list(X), list(Y)]]) + '\n')
            if player2_actions > 0:
                Y = dnp[shape[0]:shape[0] + shape[1]]
                p2 = player1payoffs.copy()
                p2 = p2.reshape((1, 2, shape[0], shape[1]))
                X = p2.transpose((0, 1, 3, 2))[:, [1, 0], :, :]
                X = X.reshape((2 * shape[0] * shape[1]))
                shape = [shape[1], shape[0]]
                f.write('\t'.join([str(i) for i in [title, shape, list(X), list(Y)]]) + '\n')


def filter_games(pool, shape=[3, 3]):
    """
    Return a subset of 'pool' containing only games that match 'shape'.
    """
    if isinstance(shape, list):
        wps = [wp for wp in pool.weighted_profiles
               if [len(pl.strategies) for pl in wp.game.players] == shape]
    elif isinstance(shape, int):
        wps = [wp for wp in pool.weighted_profiles if
               max([len(pl.strategies) for pl in wp.game.players]) < shape]
    return bogota.datapool.DataPool(wps)


def encode_game(g, shape=None, target=None, normalize=False,
                separate_players=False):
    """
    Return a vector representing the payoffs of 'g' in a predictable order,
    packed into vector of length `len(max_shape) * reduce(mul, max_shape)`.

    If 'normalize' is `True`, then payoffs will be normalized such that the
    maximum payoff in the game is 1.0.  If it is a number, payoffs will be
    normalized such that that number would be normalized to 1.0.

    If 'separate_players' is `True` returns the payoffs for each player on a
    separate dimension. Currently only supports 2 players.
    """
    if shape is None:
        shape = [len(pl.strategies) for pl in g.players]
    else:
        assert [len(pl.strategies)
                for pl in g.players] == shape, \
            "%s does not have shape %s" % (str(g), shape)

    if isinstance(normalize, list):
        Xs = [encode_game(g, shape, None, nrm) for nrm in normalize]
        X = np.zeros(reduce(add, (x.shape[0] for x in Xs)))
        s = 0
        for x in Xs:
            X[s:s + x.shape[0]] = x
            s += x.shape[0]
        return X

    if separate_players:
        sz = (2, reduce(mul, shape) * len(shape) / 2)
    else:
        sz = reduce(mul, shape) * len(shape)

    if target is None:
        target = np.zeros(sz)
    else:
        assert target.shape == tuple([sz])

    # Copy
    j = 0
    for pl_id, pl in enumerate(g.players):
        for a in action_profiles(g):
            target[j] = a.payoff(pl)
            j += 1

    # Normalize
    if isinstance(normalize, float):
        target /= normalize
    elif normalize:
        target /= target.max()

    return target


def encode_pool_list(pool, normalize_by=None):
    """
    Return a matrix X of `len(shape) * reduce(mul, shape)` columns and `pool.n`
    rows, encoding the payoffs of all the games in 'pool', and a vector Y, with
    `pool.n` rows, of integer "labels" representing which action was played for
    each example.  The labels run from 0 to `reduce(add, shape)-1`.
    """
    output_games = {}

    Xmax = 0.0
    for wp in pool:
        # Copy the game to a block of X
        shape = [len(pl.strategies) for pl in wp.game.players]
        idx = tuple(shape)
        dnp = np.array(wp.denormalized_profile())
        player1_actions = sum([dnp[i] for i in range(shape[0])])
        player2_actions = wp.n - player1_actions

        player1payoffs = encode_game(wp.game, shape,
                                     normalize=(normalize_by == 'game' or
                                                normalize_by is True))

        Y = dnp[0:shape[0]]
        X = player1payoffs

        Xmax = max([Xmax] + list(player1payoffs))
        if idx in output_games:
            output_games[idx][0] = np.vstack((output_games[idx][0], X))
            output_games[idx][1] = np.vstack((output_games[idx][1], Y))
        else:
            X = X.reshape((1, X.shape[0]))
            Y = Y.reshape((1, Y.shape[0]))
            output_games[idx] = [X, Y]

        if player2_actions > 0:
            Y = dnp[shape[0]:shape[0] + shape[1]]
            p2 = player1payoffs.copy()
            p2 = p2.reshape((1, 2, shape[0], shape[1]))
            X = p2.transpose((0, 1, 3, 2))[:, [1, 0], :, :]
            X = X.reshape((2 * shape[0] * shape[1]))
            idx = tuple([shape[1], shape[0]])
            if idx in output_games:
                output_games[idx][0] = np.vstack((output_games[idx][0], X))
                output_games[idx][1] = np.vstack((output_games[idx][1], Y))
            else:
                X = X.reshape((1, X.shape[0]))
                Y = Y.reshape((1, Y.shape[0]))
                output_games[idx] = [X, Y]

    if normalize_by == 'pool':
        for idx in output_games:
            output_games[idx][0] /= Xmax

    elif isinstance(normalize_by, float):
        for idx in output_games:
            output_games[idx][0] /= normalize_by

    return output_games


data = GameData('./all9.csv', 50.)
train_data, test_data = data.train_test(0, seed=101)

# np.sum(data._data[(4, 2)][0]) / (data._data[(4, 2)][0].shape[0]*data._data[(4, 2)][0].shape[1])