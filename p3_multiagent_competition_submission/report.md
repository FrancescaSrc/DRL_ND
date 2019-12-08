
[image1]: plot_rewards.jpg

## Report for Project 3 - Collaboration and Competition - Deep Reinforcement Learning Nanodegree


This report contains the details of the code used to solve the third project in Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.  


### Description of the project and the environment 

Project: build a reinforcement learning agent that controls rackets to bounce a ball over a net within the Unity Tennis environment. The goal of each agent is to keep the ball in play.

Agent: controls a racket and moves in a continuous space

Reward: +0.1 if an agent hits the ball over the net and -0.1 if an agent lets a ball hit the ground or hits the ball out of bounds

Goal: the goal of each agent is to keep the ball in play.

Observation space:  8 variables corresponding to the position and velocity of the ball and racket. 
Each episode yields 2 (potentially different) scores, the maximum of these 2 scores is kept for average metrics.
This single **max score** for each episode averaged on last 100 episodes should be over 0.5.

## Final goal
The environment is considered solved, when **the average (over 100 episodes) of the max scores** is at least +0.5. 


## The model and the algorithm

The algorithm I have used is an implementation of Udacity's DDGP algorithm, Deep Deterministic Policy Gradient with Soft Updates and an Experience Replay, presented in the paper [Continuous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf).

The actor-critic architecture:
 
 Actor network:
 - input is the state size of 24 units 
 - 2 linear layers with 2 hidden layers of 500 and 300 units with a RELU activation
 - output layer: 2 units, output 2 values for the action between -1 and 1 through a tanh activation
 - batch nomalization added for both layers
 
```python
# Actor Network (w/ Target Network)
self.actor_local = Actor(state_size, action_size, random_seed).to(device)
self.actor_target = Actor(state_size, action_size, random_seed).to(device)
self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

#forward pass
 def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        return F.tanh(self.fc3(x))
```
 
 Critic network:
 - input is the state size of 24 units 
 - 2 linear layers with 3 hidden layers of 500 and 300 units with a RELU activation
 - output layer: 1 unit, a Q_value  with a RELU activation
 
 
```python
# Critic Network (w/ Target Network)
self.critic_local = Critic(state_size, action_size, random_seed).to(device)
self.critic_target = Critic(state_size, action_size, random_seed).to(device)
self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

#forward pass
def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.bn1(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)
```

### Hyperparameters tuning

Agent hyperparameters:
- BUFFER_SIZE = int(2e6)  # replay buffer size
- BATCH_SIZE = 512        # minibatch size
- GAMMA = 0.99            # discount factor
- TAU = 0.009             # for soft update of target parameters
- LR = 0.0007             # learning rate 


### Description of the DDGP algorithm

The environment generates one brain which runs the 2 agents.
One instance of the agent is created with the given hyperparameters. The same object is used for both agents since the agents are competing with the same capabilities and tasks. The instance of the agent initiates 4 neural networks: actor local, actor target, critic local, critic target. 
The DDGP contains functions to get the state (between -1 and 1) from the neural network and send it to the environment to generate the next states and rewards. The function includes a ReplayBuffer class which creates a fills a buffer store with experience tuples from both agents. When the memory reaches more than 512 units (batch size) after 20 time steps, the agents will sample from this memory and call the learn function to update the policy and value parameters using the batch of experience tuples and a discount factor gamma.
```python
# Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size and timestep % LEARN_EVERY == 0:
            for _ in range(LEARN_NUM):
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
```

The learn step will compute the Q targets, the rewards and the critic loss, then calculate the loss of the actor prediction and back propagate. After the update of the target networks, the target parameters will be copied into the local network to integrate them slowly into the local network parameters. This is done with the soft update function using a tau rate of 0.5%.
```python
# ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)   
```

At the end of each loop, the scores are added to the score array and the maximum score is saved in a new array to calculate the metrics over the last 100 episodes and print the average score. When the performance reaches the average score of +0.5 the environment is solved and the weights of the neural networks are saved.

	
### Plot of Rewards
**My environment was solved in 151 episodes!** 

![Plot of training scores][image1]


### My experiments and hyperparameters

 For this project I built up on the previous one, using the same code with little edits and some changes in the hyperparameters.
 The two agents are just doing the same tasks working together and competing at the same time. The memory of experiences is important for both agents.

 I have made the following adjustments in the hyperparameters:
 - **learning rate**: I tried different lr rates but at the end my previous project rate of 0.00078 gave the best results
 - **batch size**: a bigger batch size gave a better training, I set it to 512 and the training was quite fast even with the cpu. 
 - **memory buffer**: I double the size of the buffer 2.000.000 and this made a huge difference
 - **tau value** for soft update: changing the tau value was key, my network converged more quickly. I have started with a tau value of 0.005 and the environment was resolved in 400/500 epochs. By increasing the value to a max of 0.009, the environment was solved in 200 epoch less, this because the value allows the local network to merge with the target a bit faster. Higher values did not result in a better score.
 - **network size**: I tried to add a new layer but made no difference. I increased the number of units to 500 for the first layer and added some more batch normalization.
 - **batch normalization**: I have applied normalization on the first and second layer. 

### Ideas for Future Work

 - I have tried to implement another algorithms such as the Proximal Policy Optimization but did not succeed (yet in time), due to project deadline
Next future steps: 
 - Priorized Experience Replay, Asynchronous Actor-Critic Agents (A3C)
 - Deep Neuroevolution for Reinforcement Learning: which seems a good solution without the gradient decent problems, see https://towardsdatascience.com/reinforcement-learning-without-gradients-evolving-agents-using-genetic-algorithms-8685817d84f, see here is the original code: https://github.com/paraschopra/deepneuroevolution


