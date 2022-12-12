from utils_dqn import LunarLanderAgent
import gym
import numpy as np

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    # env.seed(42)

    result_folder = 'nn_parameters'

    alpha = 0.001
    gamma = 0.99
    eps_start = 1.0
    eps_decay = 0.999
    eps_min = 0.01
    n_episodes = 100
    buffer_size = 1e5
    batch_size = 64
    update_goal_freq = 4
    early_stop_score = 200
    max_steps = 1000

    n_hidden_1 = 64
    n_hidden_2 = 64

    n_iter = 100

    # tune alpha
    alphas = np.array([0.01, 0.005, 0.001, 0.0005])
    train_scores = np.zeros((alphas.shape[0], 2, n_episodes), dtype=np.float)
    test_scores = np.zeros((alphas.shape[0], n_iter), dtype=np.float)
    i = 0
    for alpha_tune in alphas:
        agent = LunarLanderAgent(
            env, alpha_tune, gamma, eps_start,
            eps_decay, eps_min, n_episodes,
            buffer_size, batch_size, update_goal_freq,
            early_stop_score, max_steps
        )
        agent.init_models(n_hidden_1, n_hidden_2)
        filename = '{}/alpha_{}.pth'.format(result_folder, i+1)
        tr_scores = agent.train()
        agent.save_dict(filename)
        tt_scores = agent.play(filename, n_iter=n_iter)
        train_scores[i] = tr_scores
        test_scores[i] = tt_scores
        i += 1
    train_scores.dump('alpha_tuning_train.pickle')
    test_scores.dump('alpha_tuning_test.pickle')

    # tune gamma
    gammas = np.array([0.5, 0.75, 0.95, 0.99, 0.999])
    train_scores = np.zeros((gammas.shape[0], 2, n_episodes), dtype=np.float)
    test_scores = np.zeros((gammas.shape[0], n_iter), dtype=np.float)
    i = 0
    for gamma_tune in gammas:
        agent = LunarLanderAgent(
            env, alpha, gamma_tune, eps_start,
            eps_decay, eps_min, n_episodes,
            buffer_size, batch_size, update_goal_freq,
            early_stop_score, max_steps
        )
        agent.init_models(n_hidden_1, n_hidden_2)
        filename = '{}/gamma_{}.pth'.format(result_folder, i+1)
        tr_scores = agent.train()
        agent.save_dict(filename)
        tt_scores = agent.play(filename, n_iter=n_iter)
        train_scores[i] = tr_scores
        test_scores[i] = tt_scores
        i += 1
    train_scores.dump('gamma_tuning_train.pickle')
    test_scores.dump('gamma_tuning_test.pickle')

    # tune eps_decay
    decays = np.array([0.999, 0.99, 0.95, 0, 75, 0.5])
    train_scores = np.zeros((decays.shape[0], 2, n_episodes), dtype=np.float)
    test_scores = np.zeros((decays.shape[0], n_iter), dtype=np.float)
    i = 0
    for decay_tune in decays:
        agent = LunarLanderAgent(
            env, alpha, gamma, eps_start,
            decay_tune, eps_min, n_episodes,
            buffer_size, batch_size, update_goal_freq,
            early_stop_score, max_steps
        )
        agent.init_models(n_hidden_1, n_hidden_2)
        filename = '{}/eps_decay_{}.pth'.format(result_folder, i+1)
        tr_scores = agent.train()
        agent.save_dict(filename)
        tt_scores = agent.play(filename, n_iter=n_iter)
        train_scores[i] = tr_scores
        test_scores[i] = tt_scores
        i += 1
    train_scores.dump('eps_decay_tuning_train.pickle')
    test_scores.dump('eps_decay_tuning_test.pickle')

    # tune eps_min
    eps_mins = np.array([0.05, 0.01, 0.005, 0.001])
    train_scores = np.zeros((eps_mins.shape[0], 2, n_episodes), dtype=np.float)
    test_scores = np.zeros((eps_mins.shape[0], n_iter), dtype=np.float)
    i = 0
    for eps_min_tune in eps_mins:
        agent = LunarLanderAgent(
            env, alpha, gamma, eps_start,
            eps_decay, eps_min_tune, n_episodes,
            buffer_size, batch_size, update_goal_freq,
            early_stop_score, max_steps
        )
        agent.init_models(n_hidden_1, n_hidden_2)
        filename = '{}/eps_min_{}.pth'.format(result_folder, i+1)
        tr_scores = agent.train()
        agent.save_dict(filename)
        tt_scores = agent.play(filename, n_iter=n_iter)
        train_scores[i] = tr_scores
        test_scores[i] = tt_scores
        i += 1
    train_scores.dump('eps_min_tuning_train.pickle')
    test_scores.dump('eps_min_tuning_test.pickle')

    # tune network size
    hidden_szs = [(32, 32), (64, 32), (64, 64), (128, 64)]
    train_scores = np.zeros((len(hidden_szs), 2, n_episodes), dtype=np.float)
    test_scores = np.zeros((len(hidden_szs), n_iter), dtype=np.float)
    i = 0
    for hidden_sz_tune in hidden_szs:
        agent = LunarLanderAgent(
            env, alpha, gamma, eps_start,
            eps_decay, eps_min, n_episodes,
            buffer_size, batch_size, update_goal_freq,
            early_stop_score, max_steps
        )
        agent.init_models(hidden_sz_tune[0], hidden_sz_tune[1])
        filename = '{}/hidden_sz_{}.pth'.format(result_folder, i+1)
        tr_scores = agent.train()
        agent.save_dict(filename)
        tt_scores = agent.play(filename, n_iter=n_iter)
        train_scores[i] = tr_scores
        test_scores[i] = tt_scores
        i += 1
    train_scores.dump('hidden_sz_tuning_train.pickle')
    test_scores.dump('hidden_sz_tuning_test.pickle')

    env.close()
