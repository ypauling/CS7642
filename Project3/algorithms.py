from soccer_game import SoccerGame, get_state_index, MAXINDEX, NACTIONS
from soccer_game import encode_action_combination, decode_action_combination
import numpy as np
import matplotlib.pyplot as plt
from cvxopt.modeling import variable, op
from cvxopt.solvers import options


def Q_agent(start_coords, start_ball_pos, seed=42):
    niters = 100000
    eps_start = 0.9
    eps_min = 0.01
    eps_decay = (eps_start - eps_min) / niters
    alpha_start = 0.9
    alpha_min = 0.001
    alpha_decay = (alpha_start - alpha_min) / niters
    gamma = 0.9

    Qa = np.zeros((MAXINDEX, NACTIONS), dtype=np.float)
    Qb = np.zeros((MAXINDEX, NACTIONS), dtype=np.float)

    np.random.seed(seed)
    game = SoccerGame(start_coords, start_ball_pos)
    game.restart()
    done = False

    eps = eps_start
    alpha = alpha_start
    record_ind = (53, 1)
    record_q = Qa[record_ind]
    error_values = []
    iter_values = []

    for i in range(niters):
        if (i + 1) % 100 == 0:
            print('Iteration {:>6d}'.format(i+1))
        if done:
            game.restart()
            done = False

        cstate = get_state_index(game.get_coords(), game.get_ball_pos())
        if np.random.random() < eps:
            actA = np.random.choice(NACTIONS)
        else:
            actA = np.argmax(Qa[cstate])
        if np.random.random() < eps:
            actB = np.random.choice(NACTIONS)
        else:
            actB = np.argmax(Qb[cstate])

        new_coords, new_ball_pos, rwdA, rwdB, done = game.step(actA, actB)
        nstate = get_state_index(new_coords, new_ball_pos)

        Qa[cstate, actA] = (1 - alpha) * Qa[cstate, actA] + alpha * \
            ((1 - gamma) * rwdA + gamma * np.max(Qa[nstate]))
        Qb[cstate, actB] = (1 - alpha) * Qb[cstate, actB] + alpha * \
            ((1 - gamma) * rwdB + gamma * np.max(Qb[nstate]))

        eps = max(eps_min, eps - eps_decay)
        alpha = max(alpha_min, alpha - alpha_decay)

        if (cstate, actA) == record_ind and actB == 4:
            # print('Here')
            error_values.append(abs(Qa[record_ind] - record_q))
            iter_values.append(i)
            record_q = Qa[record_ind]

    return error_values, iter_values


def FriendQ_agent(start_coords, start_ball_pos, seed=42):
    niters = 100000
    eps_start = 1.0
    # eps_min = 0.01
    # eps_decay = (eps_start - eps_min) / niters
    alpha_start = 0.9
    alpha_min = 0.001
    alpha_decay = (alpha_start - alpha_min) / niters
    gamma = 0.9

    Q = np.zeros((MAXINDEX, NACTIONS * NACTIONS), dtype=np.float)

    np.random.seed(seed)
    game = SoccerGame(start_coords, start_ball_pos)
    game.restart()
    done = False

    eps = eps_start
    alpha = alpha_start
    record_ind = (53, encode_action_combination(1, 4, NACTIONS))
    record_q = Q[record_ind]
    error_values = []
    iter_values = []

    for i in range(niters):
        if (i + 1) % 100 == 0:
            print('Iteration {:>6d}'.format(i+1))
        if done:
            game.restart()
            done = False

        cstate = get_state_index(game.get_coords(), game.get_ball_pos())
        if np.random.random() < eps:
            actA = np.random.choice(NACTIONS)
            actB = np.random.choice(NACTIONS)
            act_comb = encode_action_combination(actA, actB, NACTIONS)
        else:
            act_comb = np.argmax(Q[cstate])
            actA, actB = decode_action_combination(act_comb, NACTIONS)

        new_coords, new_ball_pos, rwdA, rwdB, done = game.step(actA, actB)
        nstate = get_state_index(new_coords, new_ball_pos)

        Q[cstate, act_comb] = (1 - alpha) * Q[cstate, act_comb] + alpha * \
            ((1 - gamma) * rwdA + gamma * np.max(Q[nstate]))

        # eps = max(eps_min, eps - eps_decay)
        alpha = max(alpha_min, alpha - alpha_decay)

        if (cstate, act_comb) == record_ind:
            # print('Here')
            error_values.append(abs(Q[record_ind] - record_q))
            iter_values.append(i)
            record_q = Q[record_ind]

    return error_values, iter_values


def FoeQ_agent(start_coords, start_ball_pos, seed=42):
    niters = 100000
    eps_start = 1.0
    # eps_min = 0.01
    # eps_decay = (eps_start - eps_min) / niters
    alpha_start = 0.9
    alpha_min = 0.001
    alpha_decay = (alpha_start - alpha_min) / niters
    gamma = 0.9

    Q = np.zeros((MAXINDEX, NACTIONS * NACTIONS), dtype=np.float)

    np.random.seed(seed)
    game = SoccerGame(start_coords, start_ball_pos)
    game.restart()
    done = False

    eps = eps_start
    alpha = alpha_start
    record_ind = (53, encode_action_combination(1, 4, NACTIONS))
    record_q = Q[record_ind]
    error_values = []
    iter_values = []
    options['show_progress'] = False

    for i in range(niters):
        if (i + 1) % 100 == 0:
            print('Iteration {:>6d}'.format(i+1))
        if done:
            game.restart()
            done = False

        cstate = get_state_index(game.get_coords(), game.get_ball_pos())
        if np.random.random() < eps:
            actA = np.random.choice(NACTIONS)
            actB = np.random.choice(NACTIONS)
            act_comb = encode_action_combination(actA, actB, NACTIONS)
        else:
            act_comb = np.argmax(Q[cstate])
            actA, actB = decode_action_combination(act_comb, NACTIONS)

        new_coords, new_ball_pos, rwdA, rwdB, done = game.step(actA, actB)
        nstate = get_state_index(new_coords, new_ball_pos)

        thetas = []
        for _ in range(NACTIONS):
            thetas.append(variable())

        # all prob >= 0
        conditions = []
        for k in range(NACTIONS):
            conditions.append((thetas[k] >= 0.))
        # probs sum to 1.0
        total_prob = sum(thetas)
        conditions.append((total_prob == 1.0))

        # the max-min problem
        v = variable()
        for j in range(NACTIONS):
            c = 0.
            for k in range(NACTIONS):
                coef = Q[nstate, encode_action_combination(k, j, NACTIONS)]
                # fairly interesting, cvxopt does not recognize np.float
                c += float(coef) * thetas[k]
            conditions.append((c >= v))

        LP = op(-v, conditions)
        LP.solve()

        Q[cstate, act_comb] = (1 - alpha) * Q[cstate, act_comb] + alpha * \
            ((1 - gamma) * rwdA + gamma * v.value[0])

        # eps = max(eps_min, eps - eps_decay)
        alpha = max(alpha_min, alpha - alpha_decay)

        if (cstate, act_comb) == record_ind:
            # print('Here')
            error_values.append(abs(Q[record_ind] - record_q))
            iter_values.append(i)
            record_q = Q[record_ind]

    return error_values, iter_values


def CEQ_agent(start_coords, start_ball_pos, seed=42):
    niters = 100000
    eps_start = 1.0
    # eps_min = 0.01
    # eps_decay = (eps_start - eps_min) / niters
    alpha_start = 0.9
    alpha_min = 0.001
    alpha_decay = (alpha_start - alpha_min) / niters
    gamma = 0.9

    Qa = np.zeros((MAXINDEX, NACTIONS * NACTIONS), dtype=np.float)
    Qb = np.zeros((MAXINDEX, NACTIONS * NACTIONS), dtype=np.float)

    np.random.seed(seed)
    game = SoccerGame(start_coords, start_ball_pos)
    game.restart()
    done = False

    eps = eps_start
    alpha = alpha_start
    record_ind = (53, encode_action_combination(1, 4, NACTIONS))
    record_q = Qa[record_ind]
    error_values = []
    iter_values = []
    options['show_progress'] = False

    Va = 0.
    Vb = 0.

    for i in range(niters):
        if (i + 1) % 100 == 0:
            print('Iteration {:>6d}'.format(i+1))
        if done:
            game.restart()
            done = False

        cstate = get_state_index(game.get_coords(), game.get_ball_pos())
        if np.random.random() < eps:
            actA = np.random.choice(NACTIONS)
            actB = np.random.choice(NACTIONS)
            act_comb = encode_action_combination(actA, actB, NACTIONS)
        else:
            act_comb = np.argmax(Qa[cstate])
            actA, actB = decode_action_combination(act_comb, NACTIONS)

        new_coords, new_ball_pos, rwdA, rwdB, done = game.step(actA, actB)
        nstate = get_state_index(new_coords, new_ball_pos)

        thetas = []
        for _ in range(NACTIONS * NACTIONS):
            thetas.append(variable())

        # all prob >= 0
        conditions = list()
        for k in range(NACTIONS * NACTIONS):
            conditions.append((thetas[k] >= 0.))
        # probs sum to 1.0
        total_prob = sum(thetas)
        conditions.append((total_prob == 1.0))

        # you cannot take better actions
        v = variable()

        for k in range(NACTIONS):
            tmp1 = 0.
            for s in range(NACTIONS):
                tmp_comb = encode_action_combination(k, s, NACTIONS)
                tmp1 += float(Qa[nstate, tmp_comb]) * thetas[tmp_comb]

            for m in range(NACTIONS):
                tmp2 = 0.
                if m == k:
                    continue
                for l in range(NACTIONS):
                    tmp_comb_1 = encode_action_combination(k, l, NACTIONS)
                    tmp_comb_2 = encode_action_combination(m, l, NACTIONS)
                    tmp2 += float(Qa[nstate, tmp_comb_2]) * thetas[tmp_comb_1]
                conditions.append((tmp1 >= tmp2))

        for k in range(NACTIONS):
            tmp1 = 0.
            for s in range(NACTIONS):
                tmp_comb = encode_action_combination(s, k, NACTIONS)
                tmp1 += float(Qb[nstate, tmp_comb]) * thetas[tmp_comb]

            for m in range(NACTIONS):
                tmp2 = 0.
                if m == k:
                    continue
                for l in range(NACTIONS):
                    tmp_comb_1 = encode_action_combination(l, k, NACTIONS)
                    tmp_comb_2 = encode_action_combination(l, m, NACTIONS)
                    tmp2 += float(Qb[nstate, tmp_comb_2]) * thetas[tmp_comb_1]
                conditions.append((tmp1 >= tmp2))

        total_return = 0.
        for k in range(NACTIONS):
            for s in range(NACTIONS):
                tmp_comb = encode_action_combination(k, s, NACTIONS)
                total_return += thetas[tmp_comb] * float(Qa[nstate, tmp_comb])
                total_return += thetas[tmp_comb] * float(Qb[nstate, tmp_comb])

        conditions.append((v == total_return))

        LP = op(-v, conditions)
        LP.solve()

        if LP.status == 'optimal':
            tmp1 = 0.
            for k in range(NACTIONS):
                for s in range(NACTIONS):
                    tmp_comb = encode_action_combination(k, s, NACTIONS)
                    tmp1 += thetas[tmp_comb].value[0] * Qa[nstate, tmp_comb]

            tmp2 = 0.
            for k in range(NACTIONS):
                for s in range(NACTIONS):
                    tmp_comb = encode_action_combination(k, s, NACTIONS)
                    tmp2 += thetas[tmp_comb].value[0] * Qb[nstate, tmp_comb]

            Va = tmp1
            Vb = tmp2

        Qa[cstate, act_comb] = (1 - alpha) * Qa[cstate, act_comb] + alpha * \
            ((1 - gamma) * rwdA + gamma * Va)
        Qb[cstate, act_comb] = (1 - alpha) * Qb[cstate, act_comb] + alpha * \
            ((1 - gamma) * rwdB + gamma * Vb)

        # eps = max(eps_min, eps - eps_decay)
        alpha = max(alpha_min, alpha - alpha_decay)

        if (cstate, act_comb) == record_ind:
            # print('Here')
            error_values.append(abs(Qa[record_ind] - record_q))
            iter_values.append(i)
            record_q = Qa[record_ind]

    return error_values, iter_values


if __name__ == '__main__':
    start_coords = np.array([[2, 1], [1, 1]], dtype=np.int)
    start_ball_pos = np.array([0, 1], dtype=np.int)

    errs, iters = CEQ_agent(start_coords, start_ball_pos)
    plt.plot(iters, errs)
    plt.ylim((0.0, 0.5))
    plt.show()
