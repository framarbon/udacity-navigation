import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQN():
    def __init__(self, env, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        """Deep Q-Learning.

        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        self.env = env
        self.eps_start = eps_start                    # initialize epsilon
        self.eps_end=eps_end 
        self.eps_decay =eps_decay
        self.n_episodes = n_episodes
        self.max_t = max_t
       
    def setup(self):
        brain_name = self.env.brain_names[0]
        brain = self.env.brains[brain_name]
        env_info = self.env.reset(train_mode=True)[brain_name]
        agent = self.Agent(state_size=len(env_info.vector_observations[0]), action_size=brain.vector_action_space_size, seed=0)
        return brain_name, env_info, agent, self.eps_start
    
    def run(self):
        brain_name, env_info, agent, eps = self.setup()
        
        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        for i_episode in range(1, self.n_episodes+1):
            env_info = self.env.reset(train_mode=True)[brain_name] # reset the environment
            state = env_info.vector_observations[0]            # get the current state
            score = 0
            for t in range(self.max_t):
                action = agent.act(state, eps)                 # select an action
                env_info = self.env.step(action)[brain_name]        # send the action to the environment
                next_state = env_info.vector_observations[0]   # get the next state
                reward = env_info.rewards[0]                   # get the reward
                done = env_info.local_done[0]                  # see if episode has finished
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break 
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            eps = max(self.eps_end, self.eps_decay*eps) # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window)>=13.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
                torch.save(agent.qnetwork_local.state_dict(), 'model.pt')
                break
        return scores

    class Agent():
        """Interacts with and learns from the environment."""

        def __init__(self, state_size, action_size, seed):
            """Initialize an Agent object.

            Params
            ======
                state_size (int): dimension of each state
                action_size (int): dimension of each action
                seed (int): random seed
            """
            self.state_size = state_size
            self.action_size = action_size
            self.seed = random.seed(seed)

            # Q-Network
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

            # Replay memory
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
            # Initialize time step (for updating every UPDATE_EVERY steps)
            self.t_step = 0

        def step(self, state, action, reward, next_state, done):
            # Save experience in replay memory
            self.memory.add(state, action, reward, next_state, done)

            # Learn every UPDATE_EVERY time steps.
            self.t_step = (self.t_step + 1) % UPDATE_EVERY
            if self.t_step == 0:
                # If enough samples are available in memory, get random subset and learn
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

        def act(self, state, eps=0.):
            """Returns actions for given state as per current policy.

            Params
            ======
                state (array_like): current state
                eps (float): epsilon, for epsilon-greedy action selection
            """
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()

            # Epsilon-greedy action selection
            if random.random() > eps:
                return np.argmax(action_values.cpu().data.numpy())
            else:
                return random.choice(np.arange(self.action_size))

        def compute_loss(self, experiences, gamma):
            states, actions, rewards, next_states, dones = experiences
            Q_target_next = self.qnetwork_target(next_states).detach().max(dim=1)[0].unsqueeze(1)
            Q_expected= self.qnetwork_local(states).gather(1, actions)

            return F.mse_loss(Q_expected, rewards + gamma * Q_target_next * (1 - dones))

        def learn(self, experiences, gamma):
            """Update value parameters using given batch of experience tuples.

            Params
            ======
                experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
                gamma (float): discount factor
            """
            ## TODO: compute and minimize the loss
            "*** YOUR CODE HERE ***"
            loss = self.compute_loss(experiences, gamma)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # ------------------- update target network ------------------- #
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

        def soft_update(self, local_model, target_model, tau):
            """Soft update model parameters.
            θ_target = τ*θ_local + (1 - τ)*θ_target

            Params
            ======
                local_model (PyTorch model): weights will be copied from
                target_model (PyTorch model): weights will be copied to
                tau (float): interpolation parameter 
            """
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)