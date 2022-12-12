import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import normalize
from torch.distributions import Categorical
from collections import deque


class MLP(nn.Module):

    def __init__(self, n_state_dims, n_hidden, n_actions):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_state_dims, n_hidden, bias=True)
        self.fc2 = nn.Linear(n_hidden, n_hidden, bias=True)
        self.fc3 = nn.Linear(n_hidden, n_actions, bias=True)

    def forward(self, x):
        x = normalize(x, dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class PolicyGradientAgent:

    def __init__(self, env, alpha, gamma,
                 batch_size, n_epochs,
                 n_hidden, beta):
        self.env = env
        self.n_state_dims = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

        self.alpha = alpha
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_hidden = n_hidden
        self.beta = beta

    def init_models(self):
        self.network = MLP(
            self.n_state_dims,
            self.n_hidden,
            self.n_actions
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.alpha)

    def discounted_reward_to_go(self, rewards):
        ret_rewards = np.empty_like(rewards, dtype=np.float)
        ret_rewards[-1] = rewards[-1]
        for i in reversed(range(0, rewards.shape[0]-1)):
            ret_rewards[i] = self.gamma * ret_rewards[i+1] + rewards[i]

        return ret_rewards

    def episode_step(self, render_or_not):
        state = self.env.reset()

        ep_actions = torch.empty(size=(0, ), dtype=torch.long)
        ep_logits = torch.empty(size=(0, self.n_actions), dtype=torch.float)
        ep_rewards = np.empty(shape=(0, ), dtype=np.float)
        ep_vs = np.empty(shape=(0, ), dtype=np.float)

        # ep_counter = 0
        # max_episodes = 1000
        done = False
        while True:
            if render_or_not:
                self.env.render()

            state_tensor = torch.tensor(state).float().unsqueeze(0)
            logits = self.network(state_tensor)
            ep_logits = torch.cat((ep_logits, logits), dim=0)

            action = Categorical(logits=logits).sample()
            ep_actions = torch.cat((ep_actions, action), dim=0)

            state, reward, done, _ = self.env.step(action.cpu().item())
            ep_rewards = np.concatenate((ep_rewards, [reward]), axis=0)

            mean_reward_so_far = np.mean(ep_rewards, axis=0, keepdims=True)
            ep_vs = np.concatenate((ep_vs, mean_reward_so_far), axis=0)
            # ep_counter += 1

            if done:  # or ep_counter > max_episodes:
                ep_rewards_to_go = self.discounted_reward_to_go(ep_rewards)
                ep_rewards_to_go -= ep_vs
                ep_log_prob = Categorical(
                    logits=ep_logits).log_prob(ep_actions)
                ep_rtg_tensor = torch.tensor(ep_rewards_to_go).float()
                ep_weighted_log_prob = ep_log_prob * ep_rtg_tensor
                sum_weighted_log_prob = torch.sum(
                    ep_weighted_log_prob).unsqueeze(0)

                final_rewards = np.sum(ep_rewards, axis=0)

                return sum_weighted_log_prob, ep_logits, final_rewards

    def calculate_loss(self, batch_logits, batch_weighted_log_prob):

        loss = -1 * torch.mean(batch_weighted_log_prob)
        entropy = Categorical(logits=batch_logits).entropy()
        loss += -1 * self.beta * torch.mean(entropy, dim=0)

        return loss

    def train(self, filename):
        batch_logits = torch.empty(size=(0, self.n_actions))
        batch_weighted_log_prob = torch.empty(size=(0,))
        all_rewards = deque([], maxlen=100)
        scores = []
        avg_scores = []
        max_score = -10000.

        for epoch in range(self.n_epochs):
            render_or_not = False
            for _ in range(self.batch_size):
                sum_weighted_log_prob, ep_logits, final_rewards = \
                    self.episode_step(render_or_not)
                batch_logits = torch.cat((batch_logits, ep_logits), dim=0)
                batch_weighted_log_prob = torch.cat((
                    batch_weighted_log_prob,
                    sum_weighted_log_prob
                ), dim=0)
                all_rewards.append(final_rewards)
                scores.append(final_rewards)
                avg_scores.append(np.mean(all_rewards))
                render_or_not = False

            if avg_scores[-1] > max_score:
                max_score = avg_scores[-1]
                torch.save(self.network.state_dict(), filename)

            loss = self.calculate_loss(batch_logits, batch_weighted_log_prob)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_logits = torch.empty(size=(0, self.n_actions))
            batch_weighted_log_prob = torch.empty(size=(0,))

        return np.array([scores, avg_scores], dtype=np.float)

    def play(self, filename, n_iter=10):
        self.network.load_state_dict(torch.load(filename))
        scores = []

        for _ in range(n_iter):
            score = 0.
            state = self.env.reset()
            done = False
            while True:
                # self.env.render()
                logits = self.network(
                    torch.from_numpy(state).float().unsqueeze(0))
                action = Categorical(logits=logits).sample().cpu().item()
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                state = next_state
                if done:
                    break
            scores.append(score)

        return np.array(scores, dtype=np.float)
