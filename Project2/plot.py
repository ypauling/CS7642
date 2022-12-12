import matplotlib.pyplot as plt
import numpy as np


def check_plot_test(filename):
    data = np.load(filename, allow_pickle=True)
    x = range(data.shape[1])
    colors = ['blue', 'red', 'green', 'purple', 'yellow', 'black']

    for i in range(data.shape[0]):
        plt.plot(x, data[i, ], color=colors[i])
    plt.show()
    plt.close()


def check_plot_train(filename):
    data = np.load(filename, allow_pickle=True)
    x = range(data.shape[2])
    colors = ['blue', 'red', 'green', 'purple', 'yellow', 'black']

    for i in range(data.shape[0]):
        plt.plot(x, data[i, 1, ], color=colors[i])
    plt.show()
    plt.close()


alpha_tr = np.load('results/alpha_train.pickle', allow_pickle=True)
gamma_tr = np.load('results/gamma_train.pickle', allow_pickle=True)
nn_size_tr = np.load('results/nn_size_train.pickle', allow_pickle=True)
batch_size_tr = np.load('results/batch_size_train.pickle', allow_pickle=True)

alpha_tt = np.load('results/alpha_test.pickle', allow_pickle=True)
gamma_tt = np.load('results/gamma_test.pickle', allow_pickle=True)
nn_size_tt = np.load('results/nn_size_test.pickle', allow_pickle=True)
batch_size_tt = np.load('results/batch_size_test.pickle', allow_pickle=True)

colors = ['blue', 'red', 'green', 'purple', 'yellow', 'black']


def create_figure1(filename):
    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(211)
    x1 = range(alpha_tr.shape[2])
    ax1.plot(x1, alpha_tr[0, 0, ], color='gray', linewidth=0.1)
    ax1.plot(x1, alpha_tr[0, 1, ], color='blue', linewidth=1)
    ax1.set_ylabel('total rewards')

    ax2 = plt.subplot(212)
    x2 = range(alpha_tt.shape[1])
    ax2.plot(x2, alpha_tt[0, ], color='blue', linewidth=1)
    ax2.set_xlabel('episodes')
    ax2.set_ylabel('total rewards')

    plt.savefig(filename, format='png')
    plt.close()


def create_other_figures(filename, data_tr, data_tt, colors,
                         para_name, label_values):
    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(211)
    x1 = range(data_tr.shape[2])
    for i in range(data_tr.shape[0]):
        ax1.plot(x1, data_tr[i, 1, ], color=colors[i],
                 label='{} = {}'.format(para_name, label_values[i]),
                 linewidth=0.5)
    ax1.legend()
    ax1.set_ylabel('total rewards')

    ax2 = plt.subplot(212)
    x2 = range(data_tt.shape[1])
    for i in range(data_tt.shape[0]):
        ax2.plot(x2, data_tt[i, ], color=colors[i], linewidth=0.5)
    ax2.set_xlabel('episodes')
    ax2.set_ylabel('total rewards')

    plt.savefig(filename, format='png')
    plt.close()


if __name__ == '__main__':
    # check_plot_test('results/alpha_test.pickle')
    # check_plot_train('results/alpha_train.pickle')

    create_figure1('figures/figure1.png')
    create_other_figures(
        'figures/figure2.png',
        alpha_tr,
        alpha_tt,
        colors,
        'alpha',
        np.array([0.01, 0.005, 0.001, 0.0005])
    )
    create_other_figures(
        'figures/figure3.png',
        gamma_tr,
        gamma_tt,
        colors,
        'gamma',
        np.array([0.999, 0.99, 0.95, 0.75, 0.5, 0.25])
    )
    create_other_figures(
        'figures/figure4.png',
        nn_size_tr,
        nn_size_tt,
        colors,
        'nn_size',
        np.array([32, 64, 128, 256])
    )
    create_other_figures(
        'figures/figure5.png',
        batch_size_tr,
        batch_size_tt,
        colors,
        'batch_size',
        np.array([16, 20, 32, 64, 100])
    )
