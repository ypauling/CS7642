from algorithms import Q_agent, FoeQ_agent, FriendQ_agent, CEQ_agent
import matplotlib.pyplot as plt
import numpy as np


def plot_function(err, iv, filename):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.subplots()
    ax.set_xlim((-0.05 * 1e4, 1e5))
    # ax.set_ylim((-0.05, 0.5))
    ax.set_xlabel('iterations')
    ax.set_ylabel('Absolute difference in Q')
    ax.plot(iv, err, color='black', linewidth=0.5)
    plt.savefig(filename, format='png')
    plt.close()


if __name__ == '__main__':
    folder = 'figures'

    start_coords = np.array([[2, 1], [1, 1]], dtype=np.int)
    start_ball_pos = np.array([0, 1], dtype=np.int)

    errs, iters = Q_agent(start_coords, start_ball_pos)
    plot_function(errs, iters, '{}/qagent.png'.format(folder))

    errs, iters = FriendQ_agent(start_coords, start_ball_pos)
    plot_function(errs, iters, '{}/friendqagent.png'.format(folder))

    errs, iters = FoeQ_agent(start_coords, start_ball_pos)
    plot_function(errs, iters, '{}/foeqagent.png'.format(folder))

    errs, iters = CEQ_agent(start_coords, start_ball_pos)
    plot_function(errs, iters, '{}/ceqagent.png'.format(folder))
