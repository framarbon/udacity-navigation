
# Navigation


## The Environment

If you haven't done it yet, run the next code cell to install a few packages.  This line will take a few minutes to run!


```python
!pip -q install ./python
```


The environment is already saved in the Workspace and can be accessed at the file path provided below.  Please run the next code cell without making any changes.


```python
from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt


# please do not modify the line below
env = UnityEnvironment(file_name="/data/Banana_Linux_NoVis/Banana.x86_64")
```

    INFO:unityagents:
    'Academy' started successfully!
    Unity Academy name: Academy
            Number of Brains: 1
            Number of External Brains : 1
            Lesson number : 0
            Reset Parameters :
    		
    Unity brain name: BananaBrain
            Number of Visual Observations (per agent): 0
            Vector Observation space type: continuous
            Vector Observation space size (per agent): 37
            Number of stacked Vector Observation: 1
            Vector Action space type: discrete
            Vector Action space size (per agent): 4
            Vector Action descriptions: , , , 


Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.


```python
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
```

## The State and Action Spaces

Run the code cell below to print some information about the environment.


```python
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)
```

    Number of agents: 1
    Number of actions: 4
    States look like: [ 1.          0.          0.          0.          0.84408134  0.          0.
      1.          0.          0.0748472   0.          1.          0.          0.
      0.25755     1.          0.          0.          0.          0.74177343
      0.          1.          0.          0.          0.25854847  0.          0.
      1.          0.          0.09355672  0.          1.          0.          0.
      0.31969345  0.          0.        ]
    States have length: 37


## The RL Agent

The agent can be found in the file `dqn_agent.py`, which implements a standard [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) agent that uses an Adam optimizer. the choice of hyperparameters can be fond at the beginning of the file.

- buffer size = 1e5
- batch size = 64
- gamma = 0.99
- tau = 1e-3
- learning rate = 5e-4
- update every 4 frames

## The Q Function

The Q function is modeled as a multilayer perceptron composed by fully connected layers and 3 hidden layers with $32\times128\times512$ rectifier units. The model is implemented in the file `model.py`

## DQN


```python
from dqn_agent import DQN

dqn = DQN(env, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995)
```


```python
scores = dqn.run()
```

    Episode 100	Average Score: 0.92
    Episode 200	Average Score: 4.11
    Episode 300	Average Score: 7.20
    Episode 400	Average Score: 10.22
    Episode 500	Average Score: 11.96
    Episode 600	Average Score: 12.66
    Episode 624	Average Score: 13.02
    Environment solved in 624 episodes!	Average Score: 13.02


### Plot of Rewards


```python
# plot the scores
def plot_rewards(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('scores.png')
    plt.show()
plot_rewards(scores)
```


![png](output_14_0.png)


## DDQN


```python
from ddqn_agent import DDQN

ddqn = DDQN(env, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995)
```


```python
scores = ddqn.run()
```

    Episode 100	Average Score: 0.81
    Episode 200	Average Score: 2.56
    Episode 300	Average Score: 6.27
    Episode 400	Average Score: 9.37
    Episode 500	Average Score: 12.07
    Episode 543	Average Score: 13.03
    Environment solved in 543 episodes!	Average Score: 13.03


### Plot of rewards


```python
plot_rewards(scores)
```


![png](output_19_0.png)


## Future Work

This environment has been solved using a standard DQN implementation. Some potential approachs to improve the performance are:
- Dueling DQN
- Prioritized Experience Replay
- Noisy Nets

These methods could also be combined for better results.
Moreover a Rainbow DQN could also be expected to outperform all the above mentioned options.


```python

```
