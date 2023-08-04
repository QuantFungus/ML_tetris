import numpy as np
import torch
import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import cv2
import tqdm as tqdm


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
    def __init__(self,env,plot_info=True,run_episodes = 10000,gamma=0.99,show_traces=False):
        # Initialize the environment, Policy approximater and optimizer
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # Initialize the policy network and optimizer
        self.policy = Policy(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.01)

        self.plot_info = plot_info
        self.run_episodes = run_episodes
        self.show_traces = show_traces
        self.gamma = gamma

    
    # Select action based on the policy function and input state
    def select_action(self, state):
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    

    # Use Policy Gradient Algorithm to train the model for given environment
    def learn(self):
        rewards_per_episode = []    # List to store rewards for each episode
        policy_losses = []  # List to store policy loss for each episode

        # for specified number of episodes
        for episode in tqdm(range(self.num_episodes)):
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

            # Compute discounted rewards
            discounts = [self.gamma**i for i in range(len(rewards))]
            discounted_rewards = [discount * reward for discount, reward in zip(discounts, rewards)]

            # Convert into Tensor and normalize discounted rewards
            discounted_rewards = torch.Tensor(discounted_rewards)
            discounted_rewards -= torch.mean(discounted_rewards)
            discounted_rewards /= torch.std(discounted_rewards)

            # Compute Policy Loss
            policy_loss = []
            for log_prob, reward in zip(log_probs, discounted_rewards):
                policy_loss.append(-log_prob * reward)
            policy_loss = torch.cat(policy_loss).sum()

            # Update Policy Network
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            # Print the reward trace if needed
            if episode % 100 == 0 and self.show_traces:
                print('Episode {}: reward = {}'.format(episode, episode_reward))

            rewards_per_episode.append(episode_reward)

        if self.plot_info == True:
            self._plot_rewards(rewards_per_episode, policy_losses)


    # Plot the reward and policy loss trace for each episode
    def _plot_info(self, rewards_per_episode, policy_losses):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(rewards_per_episode)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward per Episode')

        plt.subplot(1, 2, 2)
        plt.plot(policy_losses)
        plt.xlabel('Episode')
        plt.ylabel('Policy Loss')
        plt.title('Policy Loss per Episode')

        plt.tight_layout()
        plt.show()

    # Save the trained model into a specified place
    # Since the algorithm is policy gradient, we're just updating the policy approximation function
    # the only 'model' needed to be saved is the policy
    def save_model(self, filename):
        torch.save(self.policy.state_dict(), filename)


    # Load the trained model and apply the policy trained
    def load_model(self, filename):
        self.policy.load_state_dict(torch.load(filename))
        self.policy.eval()  # Set the policy network to evaluation mode

    # Run the trained model accordingly
    def run_model(self, max_steps=1000):
        observations = self.env.reset()
        state = np.array(observations[0])
        total_reward = 0

        for step in range(max_steps):
            action, _ = self.select_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            total_reward += reward

            if done or truncated:
                break

            state = np.array(next_state)

        return total_reward