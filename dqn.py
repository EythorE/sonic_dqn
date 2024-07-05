import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

class DqnNetwork():
    def __init__(
            self, ActionNetwork: nn.Module,
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
        self.online_q_network = ActionNetwork(
                n_outputs=n_actions,
                observation_space=observation_space
                ).to(device)
        self.target_q_network = ActionNetwork(
                n_outputs=n_actions,
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
        elif optim_params:
            self.optimizer = optim_params(self.online_q_network.parameters)

    def save(self, path):
        torch.save(self.online_q_network.state_dict(), path)

    def load(self, path):
        weights = torch.load(path)
        self.online_q_network.load_state_dict(weights)
        self.target_q_network.load_state_dict(weights)

    def copy_online_to_target(self):
        self.target_q_network.load_state_dict(self.online_q_network.state_dict())

    def q_values(self, observations):
        '''Estimate q values for each possible action
        Expects a batch of observations'''
        with torch.no_grad():
            observations = torch.from_numpy(observations).to(self.device)
            q_values = self.online_q_network(observations)
            self.last_q_values = q_values.cpu().numpy()
        return self.last_q_values 
    
    def afn_random(self, observation: None=None):
        self.last_q_values = None
        return random.randrange(self.n_actions) 

    def afn_greedy(self, observation):
        q_values = self.q_values(observation[None])
        return np.argmax(q_values[0])
        
    def afn_epsilon_greedy(self, observation, epsilon):
        if random.random() < epsilon:
            self.last_q_values = None
            return random.randrange(self.n_actions)
        else:
            q_values = self.q_values(observation[None])
            return np.argmax(q_values[0])
            
    def compute_loss(self, observation_batch, oh_actions, q_value_targets):
        # Inputs should be tensors on device
        # outputs: [bs, 10], actions: [bs], q_values [bs, 1]
        outputs = self.online_q_network(observation_batch)
        q_values = (outputs*oh_actions).sum(axis=1, keepdims=True)
        loss_val = self.loss_fn(q_values, q_value_targets)
        return loss_val

    def training_step(self, memories):
        obss, actions, rewards, next_obss, continues = memories
        obss = torch.from_numpy(obss).to(self.device).float()
        with torch.no_grad():
            # Commpute q_values = rewards + future rewards
            next_obss = torch.from_numpy(next_obss).to(self.device).float()
            next_q_values = self.target_q_network(next_obss).cpu().numpy()
            max_next_q_values = next_q_values.max(axis=1, keepdims=True)
            future_rewards = continues * self.discount_rate * max_next_q_values
            q_value_targets = rewards + future_rewards 
            
            actions = torch.from_numpy(actions).long()
            oh_actions = F.one_hot(actions, num_classes=self.n_actions).to(self.device).float()
            
        q_value_targets = torch.from_numpy(q_value_targets).to(self.device).float()
        loss = self.compute_loss(obss, oh_actions, q_value_targets)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm_(self.online_q_network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return {'loss': loss.cpu().detach(), 'grad_norm': grad_norm.cpu().detach()}
