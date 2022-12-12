# Collection of functions useful for the random walk simulation
# discussed in Sutton 1988
import numpy as np

STATES = np.array([
    [0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1]
], dtype=np.int)


def sample_run(start, rb):
    '''
    A util function for generating a single run

    Parameters:
    start: [int] the starting state
    rb: [int] right bound of the sequence, left bound is always 0

    Return: [list]
    A list of sample run consisting of state integers
    '''
    if start == 0 or start == rb:
        return [start]
    else:
        result = []
        cur = start
        result.append(cur)
        while cur != 0 and cur != rb:
            move = np.random.choice([-1, 1], replace=False)
            cur += move
            result.append(cur)
        return result


def generate_samples(nstates, nsamples, nsets):
    '''
    A util function for generating experimental runs

    Parameters:
    nstates: [num] number of states in the middle
    nsamples: [num] number of samples per training set
    nsets: [num] number of training samples

    Return: [list]
    The generated training sets
    '''
    start_state = (nstates + 1) // 2
    right_bound = nstates + 1
    samples = []
    for _ in range(nsets):
        cur_set = []
        for _ in range(nsamples):
            cur_set.append(sample_run(start_state, right_bound))
        samples.append(cur_set)

    return samples


def single_sequence_delta(W, expr_run, alpha, lamda):
    '''
    A util function to calculate deltaW for a sample run

    Parameters:
    W: [np.array] the parameter vector
    expr_run: [list] a sample run
    alpha: [num] the step parameter
    lamda: [num] the decaying parameter

    Return: [num]
    The accumulated deltaW for the sample sequence
    '''
    reward = 0
    if expr_run[-1] != 0:
        reward = 1

    sample = np.array(expr_run[:-1], dtype=np.int)
    X = STATES[sample]
    preds = np.matmul(X, W)
    preds = np.append(preds, reward)

    preds_diff = preds[1:] - preds[0:-1]

    grads = np.zeros(X.shape, dtype=np.float)
    grads[0, ] = X[0, ]
    for i in range(1, X.shape[0]):
        grads[i, ] = lamda * grads[i-1, ] + X[i, ]

    temp = alpha * np.expand_dims(preds_diff, axis=1) * grads
    deltaW = np.sum(temp, axis=0)

    return deltaW
