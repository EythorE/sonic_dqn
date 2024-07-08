import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

class DqnNetwork():
    def __init__(
            self, QNetwork: nn.Module,
            optim_params: dict,
            observation_space,
            n_actions,
            discount_rate,
            loss_fn, max_grad_norm,
            device='cpu'
        ):

        self.observation_space = observation_space
        self.n_actions = n_actions
        self.discount_rate = discount_rate
        self.loss_fn = loss_fn
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.online_q_network = QNetwork(
                n_actions=n_actions,
                observation_space=observation_space
                ).to(device)
        self.target_q_network = QNetwork(
                n_actions=n_actions,
                observation_space=observation_space
                ).to(device)
        self.last_q_values = None
        
        for param in self.target_q_network.parameters():
            param.requires_grad = False

        if isinstance(optim_params, dict):
            # Get the optimizer class from the string name
            OptimizerClass = getattr(torch.optim, optim_params.pop("optim"))
            self.optimizer = OptimizerClass(
                    self.online_q_network.parameters(),
                    **optim_params
                    # nesterov=True
                    )
        elif optim_params: # assume it's an torch.optim optimizer
            self.optimizer = optim_params(self.online_q_network.parameters)

    def save(self, path):
        torch.save(self.online_q_network.state_dict(), path)

    def load(self, path):
        weights = torch.load(path)
        self.online_q_network.load_state_dict(weights)
        self.target_q_network.load_state_dict(weights)

    def copy_online_to_target(self):
        self.target_q_network.load_state_dict(self.online_q_network.state_dict())

    def q_values(self, observations, actions):
        '''Estimate q values for each possible action
        Expects a batch of observations'''
        with torch.no_grad():
            observations = torch.from_numpy(observations).to(self.device)
            actions = torch.from_numpy(actions).to(self.device)
            q_values = self.online_q_network(observations, actions)
            self.last_q_values = q_values.cpu().numpy() # should be B,T,n_actions
        return self.last_q_values 
    
    def afn_random(self, observations: None=None, actions: None=None):
        self.last_q_values = None
        return random.randrange(self.n_actions) 

    def afn_greedy(self, observations, actions):
        observations = np.array(observations)[None]
        actions = np.array(actions)[None]
        q_values = self.q_values(observations, actions)
        return np.argmax(q_values[0,-1]) # expecting B=1 and we want the last steps estimated q_values
        
    def afn_epsilon_greedy(self, observations, actions, epsilon):
        if random.random() < epsilon:
            return self.afn_random()
        else:
            return self.afn_greedy(observations, actions)
            
    def compute_loss(self, observation_batch, actions, oh_chosen_actions, q_value_targets, mask):
        # Inputs should be tensors on device
        # outputs: [bs, Tmax, 10], actions: [bs, Tmax], q_values [bs, Tmax, 1], mask [bs, Tmax]
        # actions are the actions taken from these observations
        outputs = self.online_q_network(observation_batch, actions)
        q_values = (outputs*oh_chosen_actions).sum(axis=-1, keepdims=True)
        loss_val = self.loss_fn(q_values*mask, q_value_targets*mask)
        return loss_val

    def training_step(self, episode_memories):
        obs, actions, rewards, continues = episode_memories
        # obs (B,T,C,W,H), actions (B,T), rewards (B,T), done (B,Ti), mask (B,T)
        obs = torch.from_numpy(obs).to(self.device).float()
        with torch.no_grad():
            # Commpute q_values = rewards + future rewards,
            # future rewards estimated from the obs in the next step
            # for each step in episode for episodes in batch
            actions = torch.from_numpy(actions).to(self.device).long() # actions -> obs
            chosen_actions = actions[:,1:] # obs ->  chosen_actions -> next_obs
            next_obs = obs[:,1:] # we are feeding action -> next_obs into the qnet
            next_q_values = self.target_q_network(next_obs, chosen_actions).cpu().numpy()
            max_next_q_values = next_q_values.max(axis=-1, keepdims=True)
            
            future_rewards = continues[:,1:,None] * self.discount_rate * max_next_q_values
            assert not  np.isnan(future_rewards).flatten().any()
            q_value_targets = rewards[:,1:,None] + future_rewards 
            assert not np.isnan(q_value_targets).flatten().any()
            
            oh_chosen_actions = F.one_hot(chosen_actions, num_classes=self.n_actions).to(self.device).float()
            mask = torch.from_numpy(continues[:,:1,None]).to(self.device)
            
        q_value_targets = torch.from_numpy(q_value_targets).to(self.device).float()
        # don't have the action for the last obs,
        # we are recording obs gotten after taking action and getting reward
        loss = self.compute_loss(obs[:,:-1], actions[:,:-1], oh_chosen_actions, q_value_targets, mask) 
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm_(self.online_q_network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return {'loss': loss.cpu().detach(), 'grad_norm': grad_norm.cpu().detach()}
