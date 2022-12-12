import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import random


class SimpleNetwork(nn.Module):

    def __init__(self, state_dims, n_actions, n_hidden_1, n_hidden_2):

        super(SimpleNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dims, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class CyclicReplayMemory:

    def __init__(self, max_size):
        self.max_size = int(max_size)
        self.memories = []
        self.pos = 0
        self.transition = namedtuple(
            'transition',
            ('state', 'action', 'reward', 'next_state', 'done')
        )

    def __len__(self):
        return len(self.memories)

    def add(self, *args):
        # if memories not full, put a placeholder
        if len(self.memories) < self.max_size:
            self.memories.append(None)
        self.memories[self.pos] = self.transition(*args)
        # move the position one step ahead mod max size
        self.pos = (self.pos + 1) % self.max_size

    def sample(self, batch_size):
        samples = random.sample(self.memories, batch_size)
        if samples is None:
            print('Random sample error...')
            exit(1)

        # change all the results to numpy arrays
        states = np.vstack([s.state for s in samples])
        actions = np.vstack([s.action for s in samples])
        rewards = np.vstack([s.reward for s in samples])
        next_states = np.vstack([s.next_state for s in samples])
        dones = np.vstack([s.done for s in samples])

        return (states, actions, rewards, next_states, dones)


class LunarLanderAgent:

    def __init__(self, env, alpha, gamma, eps_start,
                 eps_decay, eps_min, n_episodes,
                 buffer_size, batch_size, update_goal_freq,
                 early_stop_score, max_steps):
        self.env = env
        self.state_dims = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.n_episodes = n_episodes
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_goal_freq = update_goal_freq
        self.early_stop_score = early_stop_score
        self.max_steps = max_steps

    def init_models(self, n_hidden_1, n_hidden_2):
        self.pred_network = SimpleNetwork(
            self.state_dims, self.n_actions,
            n_hidden_1, n_hidden_2
        )
        self.goal_network = SimpleNetwork(
            self.state_dims, self.n_actions,
            n_hidden_1, n_hidden_2
        )
        self.optimizer = optim.Adam(
            self.pred_network.parameters(), lr=self.alpha)
        self.replay_mem = CyclicReplayMemory(self.buffer_size)

    def choose_action(self, state, eps):

        # implement epsilon-greedy strat
        if np.random.rand() < eps:
            return np.random.randint(self.n_actions)
        else:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            self.pred_network.eval()
            with torch.no_grad():
                action_qs = self.pred_network(state_tensor)
            self.pred_network.train()
            retact = np.argmax(action_qs.cpu().data.numpy())
            return retact

    def update_goal_network(self):
        for gp, pp in zip(self.goal_network.parameters(),
                          self.pred_network.parameters()):
            gp.data.copy_(pp.data)

    def update_pred_network(self):
        sample = self.replay_mem.sample(self.batch_size)
        states, actions, rewards, next_states, dones = sample

        # convert everything to tensor
        cst_tensor = torch.from_numpy(states).float()
        act_tensor = torch.from_numpy(actions).long()
        rwd_tensor = torch.from_numpy(rewards).float()
        nst_tensor = torch.from_numpy(next_states).float()
        dns_tensor = torch.from_numpy(dones.astype(np.int)).float()

        action_qs = self.goal_network(nst_tensor).detach()
        max_action_qs = action_qs.max(1)[0].unsqueeze(1)

        target_values = rwd_tensor + \
            (self.gamma * max_action_qs * (1 - dns_tensor))
        result_values = self.pred_network(cst_tensor).gather(1, act_tensor)

        # actual learning
        loss = F.mse_loss(target_values, result_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        scores = []
        avg_scores = []
        last_scores = deque(maxlen=100)
        total_steps = 0
        eps = self.eps_start

        for i in range(self.n_episodes):
            # print('Start episode {}'.format(i+1))
            state = self.env.reset()
            score = 0
            done = False

            for _ in range(self.max_steps):
                action = self.choose_action(state, eps)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_mem.add(state, action, reward, next_state, done)
                score += reward
                total_steps += 1

                if len(self.replay_mem) <= self.batch_size:
                    continue

                self.update_pred_network()
                if total_steps % self.update_goal_freq == 0:
                    self.update_goal_network()

                eps = max(eps * self.eps_decay, self.eps_min)
                state = next_state

                if done:
                    break

                if score >= self.early_stop_score:
                    break

            if score != 0:
                scores.append(score)
                last_scores.append(score)
                avg_scores.append(np.mean(last_scores))
                if (i+1) % 10 == 0:
                    print('Episode {:>4d}, score {:> 3.2f}'.format(
                        i+1, np.mean(last_scores)))

        ret = np.array([scores, avg_scores], dtype=np.float)
        return ret

    def play(self, filename, n_iter=10):
        self.pred_network.load_state_dict(torch.load(filename))
        scores = []
        for _ in range(n_iter):
            state = self.env.reset()
            score = 0.
            done = False
            while True:
                # self.env.render()
                action = self.choose_action(state, 0.0)
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                state = next_state
                if done:
                    break
            # print('Final reward: {:.3f}'.format(score))
            scores.append(score)
        return np.array(scores, dtype=np.float)

    def save_dict(self, filename):
        torch.save(self.pred_network.state_dict(), filename)
