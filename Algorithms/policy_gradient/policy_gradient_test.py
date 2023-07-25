import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import time
import cv2

# Define the Policy Network
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

# Create the environment
env = gym.make('CartPole-v1',render_mode="rgb_array")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize the policy network
policy = Policy(state_dim, action_dim)

# Define the optimizer
optimizer = optim.Adam(policy.parameters(), lr=0.01)

# Function to select an action based on the policy
def select_action(state):
    state = np.array(state)
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)

# Training loop
def policy_gradient():
    num_episodes = 1000
    gamma = 0.99

    # Create a VideoWriter object
    video_writer = cv2.VideoWriter('cartpole_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (600, 400))

    # for 1000 episodes
    for episode in range(num_episodes):
        observations = env.reset()
        state = np.array(observations[0])
        episode_reward = 0
        log_probs = []
        rewards = []

        # loop through each time step in one episode
        while True:
            action, log_prob = select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            episode_reward += reward

            if done:
                break

            state = next_state
            # Visualize the frame
            frame = env.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert color channels from RGB to BGR for video
            frame = cv2.resize(frame, (600, 400))  # Resize the frame to fit the video dimensions
            video_writer.write(frame)
            

        # Compute the discounted rewards
        discounts = [gamma**i for i in range(len(rewards))]
        discounted_rewards = [discount * reward for discount, reward in zip(discounts, rewards)]

        # Convert the discounted_rewards into a Tensor
        discounted_rewards = torch.Tensor(discounted_rewards)

        # Normalize the discounted rewards        
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)

        # Calculate the loss
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        policy_loss = torch.cat(policy_loss).sum()

        # Update the policy network
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # Print the episode statistics
        if episode % 10 == 0:
            print('Episode {}: reward = {}'.format(episode, episode_reward))
    
    # Release the VideoWriter object
    video_writer.release()


# Train the policy network
policy_gradient()
