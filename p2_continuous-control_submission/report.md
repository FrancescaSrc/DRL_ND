
[image1]: assets/graph_testresults.jpg

# Report for Project 2 - Continuous Control - Deep Reinforcement Learning Nanodegree


This report contains the details of the code used to solve the second project in Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.  


# Project 2: Continuous Control

### Description of the environment 

Build a reinforcement learning agent that controls a robotic arm within Unity's Reacher environment. The goal is to get 20 robotic arms to maintain contact with a green sphere.
agent: a double-jointed arm that moves to target locations. 
reward:  +0.1 for each step that the agent's hand is in the goal location. 
goal: the agent maintains its position at the target location for as many time steps as possible.
observation space: 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. 
action: a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## Goals
solve the environment with one agent: the agent must get an average score of +30 over 100 consecutive episodes
solve the environment with 20 agents: the environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 


## The model and the algorithm

 The algorithm I have used is an implementation of Udacity's DDGP algorithm, Deep Deterministic Policy Gradient with Soft Updates and an Experience Replay, presented in the paper [Continuous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf).

 The actor-critic model:
 Actor model:
 - input is the state size of 33 units 
 - 2 linear layers with 2 hidden layers of 400 and 300 units with a RELU activation
 - output layer: 4 units, output 4 values for the action between -1 and 1 through a tanh activation
 The algorithm
 Critic network 
 - input is the state size of 33 units 
 - 2 linear layers with 3 hidden layers of 400 and 300 units with a RELU activation
 - output layer: 1 unit, a Q_value  with a RELU activation

### Hyperparameters tuning

Agent hyperparameters:
- BUFFER_SIZE = int(1e6)  # replay buffer size
- BATCH_SIZE = 256        # minibatch size
- GAMMA = 0.99            # discount factor
- TAU = 0.005             # for soft update of target parameters
- LR = 0.00078            # learning rate 


### The DDGP algorithm

	The environment generates one brain which runs 20 agents.
	One instance of the agent is created with the given hyperparameters. The same object is used for all 20 agents since the agents are exactly the same. The instance of the agent initiates 4 neural networks: actor local, actor target, critic local, critic target. 
	The act function inputs the state to the actor local network, add noise and returns a given state (between -1 and 1). 
	The action is sent to the environment which generates the next states and rewards. 
	The step function is called which will create an experience replay buffer from which to sample.
	When the memory reaches more than 512 units (batch size) after 20 time steps, the agent will sample from this memory and call the learn function to update the policy and value parameters using the batch of experience tuples and a discount factor gamma.
	The learn step will compute the Q targets, the rewards and the critic loss, then calculate the loss of the actor prediction and back propagate. After the update of the target networks, the target parameters will be copied into the local network to integrate them slowly into the local network parameters. This is done with the soft update function using a tau rate of 0.5%.
	At the end of each loop, the rewards are added to the score array and the average metrics are calculated: per episode, per 100 episodes and the performance average (moving avg) is printed. When the performance reaches the score of 30 the environment is solved and the weights are saved.

	


### Plot of Rewards
**My environment was solved in 9 episodes!** (but took at least 7.5 hours to train)

![Plot of training scores][image4]


### My experiments and hyperparameters

 The training of a single agent was disappointing and I was really surprised by its bad performace.
 Only after many hours of training I realized that the 20 agents together were performing much better.

 I have tried to tune many different hyperparameters:
 - learning rate: starting with a learning rate of 0.001, I have decrease it slowly till 0.0008 which gave the best results
 - batch size: increasing the batch size gave a better training, but it was also really slow. I set for 512 and the training lasted about 5 hours. Before that, I had tried 1024 but the nightly training did not manage to finish.
 - tau value for soft update: I have increased the tau value at 0.005 allowing the local network to merge with the target a bit faster. 
 - learning time step interval and number of learning passes: changing this values made the training slow and did not improve the results, but my attempts concerned only the single agent, I did not try to change it in the multi agent version.
 - network size: I added a new layer and tried to change the number of units but did not see any improvements, the training took a lot longer
 - batch normalization: I have applied normalization on the first layer, with momentum option set to 0. 

### Ideas for Future Work

The algorithm solves a simple game but if it were used for a more difficult task several improvements could be added:
 - a different weight initialization (which I tried to implement without success)
 - pretraining techniques to speed up the training
 - try to implement another algorithm such as the GAE Generalized Advantage, PPO, and D4PG
 -  implementing Deep Neuroevolution for Reinforcement Learning: while searching for a solution to improve my agents performance, I came this article across: https://towardsdatascience.com/reinforcement-learning-without-gradients-evolving-agents-using-genetic-algorithms-8685817d84f
 which proposes a different approch, a genetic algorithm which  here is the original code: https://github.com/paraschopra/deepneuroevolution


