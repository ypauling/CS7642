import gym
import numpy as np
from libs import PolicyGradientAgent


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    # env.seed(42)

    nn_folder = 'nn_state_dict'
    result_folder = 'results'

    alpha = 0.01
    gamma = 0.99
    batch_size = 64
    n_epochs = 200
    n_hidden = 64
    beta = 0.1

    n_iter = 100
    # tune alpha
    param_name = 'alpha'
    params = np.array([0.01, 0.005, 0.001, 0.0005])
    train_scores = np.zeros(
        (len(params), 2, n_epochs*batch_size), dtype=np.float)
    test_scores = np.zeros((len(params), n_iter), dtype=np.float)
    i = 0
    for param_tune in params:
        filename = '{}/{}_{}.pth'.format(nn_folder, param_name, i+1)
        agent = PolicyGradientAgent(
            env, param_tune, gamma,
            batch_size, n_epochs,
            n_hidden, beta
        )
        agent.init_models()
        train_scores[i] = agent.train(filename)
        test_scores[i] = agent.play(filename, n_iter=n_iter)
        i += 1

    train_scores.dump('{}/{}_train.pickle'.format(result_folder, param_name))
    test_scores.dump('{}/{}_test.pickle'.format(result_folder, param_name))

    # tune gamma
    param_name = 'gamma'
    params = np.array([0.999, 0.99, 0.95, 0.75, 0.5, 0.25])
    train_scores = np.zeros(
        (len(params), 2, n_epochs*batch_size), dtype=np.float)
    test_scores = np.zeros((len(params), n_iter), dtype=np.float)
    i = 0
    for param_tune in params:
        filename = '{}/{}_{}.pth'.format(nn_folder, param_name, i+1)
        agent = PolicyGradientAgent(
            env, alpha, param_tune,
            batch_size, n_epochs,
            n_hidden, beta
        )
        agent.init_models()
        train_scores[i] = agent.train(filename)
        test_scores[i] = agent.play(filename, n_iter=n_iter)
        i += 1

    train_scores.dump('{}/{}_train.pickle'.format(result_folder, param_name))
    test_scores.dump('{}/{}_test.pickle'.format(result_folder, param_name))

    # tune nn size
    param_name = 'nn_size'
    params = np.array([32, 64, 128, 256])
    train_scores = np.zeros(
        (len(params), 2, n_epochs*batch_size), dtype=np.float)
    test_scores = np.zeros((len(params), n_iter), dtype=np.float)
    i = 0
    for param_tune in params:
        filename = '{}/{}_{}.pth'.format(nn_folder, param_name, i+1)
        agent = PolicyGradientAgent(
            env, alpha, gamma,
            batch_size, n_epochs,
            param_tune, beta
        )
        agent.init_models()
        train_scores[i] = agent.train(filename)
        test_scores[i] = agent.play(filename, n_iter=n_iter)
        i += 1

    train_scores.dump('{}/{}_train.pickle'.format(result_folder, param_name))
    test_scores.dump('{}/{}_test.pickle'.format(result_folder, param_name))

    # tune batch_size
    param_name = 'batch_size'
    params = np.array([16, 20, 32, 64, 100])
    train_scores = np.zeros(
        (len(params), 2, n_epochs*batch_size), dtype=np.float)
    test_scores = np.zeros((len(params), n_iter), dtype=np.float)
    i = 0
    for param_tune in params:
        filename = '{}/{}_{}.pth'.format(nn_folder, param_name, i+1)
        agent = PolicyGradientAgent(
            env, alpha, gamma,
            param_tune, n_epochs*batch_size // param_tune,
            n_hidden, beta
        )
        agent.init_models()
        train_scores[i] = agent.train(filename)
        test_scores[i] = agent.play(filename, n_iter=n_iter)
        i += 1

    train_scores.dump('{}/{}_train.pickle'.format(result_folder, param_name))
    test_scores.dump('{}/{}_test.pickle'.format(result_folder, param_name))

    env.close()
