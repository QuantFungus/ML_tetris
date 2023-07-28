# Policy Gradient Algorithm with PyTorch
This repository contains an implementation of the Policy Gradient algorithm using PyTorch. The Policy Gradient algorithm is a popular reinforcement learning technique for learning the policy function of an agent to maximize the cumulative reward in an environment.

## Algorithm Description

Policy Gradient is a popular reinforcement learning method for training agents in environments with discrete or continuous action spaces. Instead of learning a value function like in Q-learning, it directly learns a policy that maps states to actions. The policy is typically parameterized by a neural network, and the agent uses gradient ascent to update the policy parameters in a direction that increases the expected cumulative rewards.

## How to Use

1.  **Initialize Policy Gradient:**

To use the Policy Gradient algorithm, you first need to initialize an instance of the `PolicyGradient` class. It takes the following parameters:

-   `env` (gym.Env): The gym environment on which the policy will be trained.
-   `plot_info` (bool, optional): If True, it will plot the reward and policy loss traces during training. Default is True.
-   `run_episodes` (int, optional): The number of episodes to run during training. Default is 10000.
-   `gamma` (float, optional): The discount factor for future rewards. Default is 0.99.
-   `show_traces` (bool, optional): If True, it will print the reward trace for every 100 episodes during training. Default is False.

```
env = gym.make('CartPole-v1') 
policy_gradient = PolicyGradient(env, plot_info=True,run_episodes=10000, gamma=0.99, show_traces=False)
```
2.  **Train the Policy:**

To train the policy network using Policy Gradient, call the `learn()` method. It will run the specified number of episodes and update the policy parameters to maximize the cumulative rewards.
```
policy_gradient.learn()
```
3.  **Save and Load Trained Model:**

After training, you can save the trained policy network to a file and load it later to use for inference or further training.

4.  **Run the Trained Policy:**

You can run the trained policy in the environment using the `run_model()` method. It will execute the policy and return the total reward obtained during the evaluation.
```
total_reward = policy_gradient.run_model()
print("Total reward:", total_reward)
```

## Example
```
# Create the environment 
env = gym.make('CartPole-v1') 

# Initialize the PolicyGradient class 
pg = PolicyGradient(env, plot_info=True, run_episodes=10000, gamma=0.99, show_traces=False) 

# Train the policy gradient algorithm 
pg.learn() # Save the trained model pg.save_model('trained_policy.pt') 

# Load the trained model 
pg.load_model('trained_policy.pt') 

# Evaluate the trained model 
total_reward = pg.run_model(max_steps=1000) 
print("Total reward:", total_reward)
```


## Tool Applied

-   PyTorch: The deep learning framework used to implement the policy network and perform gradient updates.
-   OpenAI Gym: The library used to create and interact with the reinforcement learning environments.
-   Matplotlib: Used for plotting the reward and policy loss traces during training.
-   tqdm: Used to provide a progress bar during training.

**Note**: Make sure you have the required libraries installed before running the code. You can install them using `pip install -r requirements.txt`.

Feel free to use this implementation to train your own policies in different environments and explore the fascinating world of reinforcement learning!