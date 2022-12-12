import numpy as np
import matplotlib.pyplot as plt
from utils import generate_samples, single_sequence_delta


def RMS(data, target):
    '''
    Implementation of the Root Mean Square error

    Parameters:
    data: [numpy array] predictions
    target: [numpy array] target of prediction

    Return:
    [num] the rms error
    '''

    data = np.array(data, dtype=np.float)
    target = np.array(target, dtype=np.float)

    return np.sqrt(((target - data) ** 2).mean())


def single_seq_simulation(lamdas, alphas, seed=20):

    np.random.seed(seed)
    target = np.array([1/6, 2/6, 3/6, 4/6, 5/6])
    results = np.zeros((len(lamdas), len(alphas)), dtype=np.float)
    train_sets = generate_samples(5, 10, 100)

    for i, lamda in enumerate(lamdas):
        for j, alpha in enumerate(alphas):
            print('Processing lamda = {:.3f}, alpha = {:.3f}'.format(
                lamda, alpha))
            total_err = 0.
            for tset in train_sets:
                W = np.array([0.5] * 5, dtype=np.float)
                for single_run in tset:
                    W = W + single_sequence_delta(W, single_run, alpha, lamda)
                total_err += RMS(W, target)
            results[i, j] = total_err / 100

    return results


def batch_simulation(lamdas, alphas, eps=1e-4, maxiter=10000, seed=20):

    np.random.seed(seed)
    target = np.array([1/6, 2/6, 3/6, 4/6, 5/6])
    results = np.zeros((len(lamdas), len(alphas)), dtype=np.float)
    train_sets = generate_samples(5, 10, 100)

    for i, lamda in enumerate(lamdas):
        for j, alpha in enumerate(alphas):
            print(
                'Processing lamda = {:.3f}, alpha = {:.3f}'.format(
                    lamda, alpha
                )
            )
            total_err = 0.
            for tset in train_sets:
                counter = 0
                W = np.array([0.5] * 5, dtype=np.float)
                while (counter <= maxiter):
                    dW = np.zeros(5)
                    for single_run in tset:
                        dW += single_sequence_delta(
                            W, single_run, alpha, lamda
                        )
                    W_prev = W.copy()
                    W += dW
                    eps_err = RMS(W_prev, W)
                    if eps_err < eps:
                        break
                    counter += 1
                total_err += RMS(W, target)
            print(total_err / 100.)
            results[i, j] = total_err / 100.

    return results


def draw_figure3(results, xvals, filename):
    yvals = np.min(results, axis=1)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_title('Sutton(1988) figure 3', fontsize=20)
    ax.set_xlabel('lambda', fontsize=20)
    ax.set_ylabel('ERROR USING BEST alpha', fontsize=20)
    ax.plot(np.array(xvals), yvals, color='black', marker='v', lw=3)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    plt.savefig(filename, format='png')


def draw_figure4(results, lamdas, alphas, inds, filename):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_title('Sutton(1988) figure 4', fontsize=20)
    ax.set_xlabel('alphas', fontsize=20)
    ax.set_ylabel('ERROR', fontsize=20)

    for i in inds:
        xvals = np.array(alphas)[np.where(results[i, ] < 1.)[0]]
        yvals = results[i, np.where(results[i, ] < 1.)[0]]
        ax.plot(xvals, yvals, color='black', marker='v', lw=1.5)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
    plt.savefig(filename, format='png')


def draw_figure5(results, xvals, filename):
    yvals = np.min(results, axis=1)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_title('Sutton(1988) figure 5', fontsize=20)
    ax.set_xlabel('lambda', fontsize=20)
    ax.set_ylabel('ERROR USING BEST alpha', fontsize=20)
    ax.plot(np.array(xvals), yvals, color='black', marker='v', lw=3)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    plt.savefig(filename, format='png')


if __name__ == '__main__':

    lamdas = np.linspace(0, 1., 11)
    alphas = np.linspace(0, 0.02, 11)

    A = batch_simulation(lamdas, alphas)
    print(A)
    print(np.min(A, axis=1))
    draw_figure3(A, lamdas, 'figure3.png')

    lamdas = np.linspace(0, 1., 11)
    alphas = np.linspace(0.0, 0.6, 13)
    inds = [0, 3, 8, 10]
    A = single_seq_simulation(lamdas, alphas)
    print(A)
    draw_figure4(A, lamdas, alphas, inds, 'figure4.png')
    draw_figure5(A, lamdas, 'figure5.png')
