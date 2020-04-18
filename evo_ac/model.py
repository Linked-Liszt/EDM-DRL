import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class EvoACModel(nn.Module):
    """
    implements both actor and critic in one model
    TODO: Add ability to contain lstm and RNN layers. 
    """
    def __init__(self, config_dict):
        super().__init__()
        self.evo_config = config_dict['evo_ac']
        self.net_config = config_dict['neural_net']
        
        self._init_model_layers()
        self.opt = optim.Adam(self.parameters(), lr=self.net_config['learning_rate'])


    def _init_model_layers(self):
        self.shared_net = self._add_layers(self.net_config['shared'])
        self.policy_nets = [self._add_layers(self.net_config['policy']) 
                            for _ in range(self.evo_config['pop_size'])]
        self.value_net = self._add_layers(self.net_config['value'])


    def _add_layers(self, layer_config):
        output_ml = nn.ModuleList()
        for layer in layer_config:
            output_ml.append(self._add_layer(
                layer['type'], 
                layer['params'],
                layer['kwargs']))
        return output_ml


    def extract_params(self):
        with torch.no_grad():
            extracted_parameters = []
            for individual in self.policy_nets:
                layer_params = []
                for layer in individual:
                    for name, parameter in layer.named_parameters():
                        layer_params.append(parameter.detach())
                extracted_parameters.append(layer_params)
            return extracted_parameters


    def insert_params(self, incoming_params):
        with torch.no_grad():
            for pop_idx in range(len(self.policy_nets)):
                params_idx = 0
                individual = self.policy_nets[pop_idx]
                for layer in individual:
                    state_dict = layer.state_dict()
                    for name, parameter in layer.named_parameters():
                        state_dict[name] = incoming_params[pop_idx][params_idx]
                        params_idx += 1
                    layer.load_state_dict(state_dict)


    def extract_grads(self):
        with torch.no_grad():
            extracted_grads = []
            for individual in self.policy_nets:
                layer_grads = []
                for layer in individual:
                    for name, parameter in layer.named_parameters():
                        layer_grads.append(parameter.grad.detach())
                extracted_grads.append(layer_grads)
            return extracted_grads


    def forward(self, x, pop_idx):
        shared = x
        for layer in self.shared_net:
            shared = layer(shared)

        policy = shared
        value = shared

        for layer in self.policy_nets[pop_idx]:
            policy = layer(policy)
        
        for layer in self.value_net:
            value = layer(value)


        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return policy, value


    def get_action(self, state, pop_idx):
        policy, value = self(state, pop_idx)

        action_prob = F.softmax(policy, dim=-1)
        cat = Categorical(action_prob)
        action = cat.sample()

        return action, cat.log_prob(action), cat.entropy().mean(), value
    

    def _add_layer(self, layer_type, layer_params, layer_kwargs):
        #Finds the pytorch relevant function and calls it with params and kwargs
        return getattr(nn, layer_type)(*layer_params, **layer_kwargs)