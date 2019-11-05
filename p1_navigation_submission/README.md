[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

The project goal is to train an agent to navigate (and collect bananas!) in a large, square and unknown world.  

![Trained Agent][image1]

The agent gets a reward of +1 for collecting a yellow banana, and a reward of -1 for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### How to get started

1. Download the environment from one of the links below. You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the created repository, under the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions to run the project

To set up this project and open the ipynb file, you will need Anaconda and an environment with all the dependencies. To install it, follow the instructions below. If are using Windows, you can install the python environment using the requirement.txt provided in this repository, please follow the instructions below.

Once installed all the dependencies, start the `Navigation.ipynb` to start the agent, eventually you can load the trained agent I have provided in this repo. Try it and then train your own agent!


## Installation of your environment

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	- If you encounter any issue with installing the mlagents, try installing the swing module first, then the box2d and the mlagents
	- you can also use the requirements.txt file included in this repository, run it in your environment with:
	```bash
	(env_name)$ pip install -r requirements.txt```
	or 
	```bash
	pip install -r requirements.txt```
	
	Manual installation of mlagents:
	- git clone https://github.com/Unity-Technologies/ml-agents
	- cd ml-agents
	- pip install mlagents

In some cases, you might also need to download and install swig for Windows. Swig is the abbreviation of Simplified Wrapper and Interface Generator, it can give script language such as python the ability to invoke C and C++ libraries interface method indirectly. Just follow the [instructions on swig installation on this site](https://www.dev2qa.com/how-to-install-swig-on-macos-linux-and-windows/)

Now open the `Navigation.ipynb` to get started with training your own agent!  


### General information on Reinforcement learning

Reinforcement Learning is a branch of Machine Learning where an agent outpust an action and the enviroment returns an observation or the state of a system and a reward. The goal of an agent is to determine the best action to take to maximaze its total reward. The environment is unknown to the agent. Deep RL uses non linear function approximimators to calculate the value actions on the base of the observations from the environment.
Deep learning is used to find the optimal parameters for this function approximators. Since the agent is handling raw data and the entire end-to-end pipeline, this is called pixels-to-action, the agent starts from raw data and chooses the action which maximizes its reward.
The RL agents can be trained to learn intuitive human behaviours, such as walking or knowledge gathering just by feeding them with images on the actions to take.


