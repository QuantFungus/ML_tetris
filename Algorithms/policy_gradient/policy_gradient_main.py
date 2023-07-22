import numpy as np
import torch
import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import cv2


# Define the policy approximater, using Neural Network
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)



class PolicyGradient:
    def __init__(self,env,plot_rewards=True,run_episodes = 10000,gamma=0.99):
        # Initialize the environment, Policy approximater and optimizer
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.policy = Policy(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.01)

        self.plot_rewards = plot_rewards
        self.run_episodes = run_episodes
        self.gamma = gamma

    
    def select_action(self, state):
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    

    def policy_gradient(self):
        rewards_per_episode = []

        for episode in range(self.num_episodes):
            observations = self.env.reset()
            state = np.array(observations[0])
            episode_reward = 0
            log_probs = []
            rewards = []

            while True:
                action, log_prob = self.select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)

                log_probs.append(log_prob)
                rewards.append(reward)
                episode_reward += reward

                if done or truncated:
                    break

                state = next_state

            discounts = [self.gamma**i for i in range(len(rewards))]
            discounted_rewards = [discount * reward for discount, reward in zip(discounts, rewards)]

            discounted_rewards = torch.Tensor(discounted_rewards)
            discounted_rewards -= torch.mean(discounted_rewards)
            discounted_rewards /= torch.std(discounted_rewards)

            policy_loss = []
            for log_prob, reward in zip(log_probs, discounted_rewards):
                policy_loss.append(-log_prob * reward)
            policy_loss = torch.cat(policy_loss).sum()

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            if episode % 10 == 0:
                print('Episode {}: reward = {}'.format(episode, episode_reward))

            rewards_per_episode.append(episode_reward)

        if self.plot_rewards == True:
            self._plot_rewards(rewards_per_episode)


    def _plot_rewards(self, rewards_per_episode):
        plt.plot(rewards_per_episode)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward per Episode')
        plt.show()