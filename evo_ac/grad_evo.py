import copy
import torch
import random
import numpy as np
from itertools import combinations


class EvoACEvoAlg(object):
    def __init__(self, config_dict):
        self.evo_config = config_dict['evo_ac']
        self.net_config = config_dict['neural_net']

        #CONSTANTS
        self.num_mutate = self.evo_config['recomb_nums'] # [4,3,2,1]
        self.learning_rate = self.evo_config['learning_rate'] #1e-3
        self.lr_decay = self.evo_config['lr_decay']
        self.mut_scale = self.evo_config['mut_scale'] #0.5

        num_children = 0
        if self.evo_config['hold_elite']:
            num_children += 1
        num_children += sum(self.num_mutate)
        num_children += self.evo_config["mate_num"]

        if num_children != self.evo_config['pop_size']:
            raise RuntimeError(f"Number children ({num_children}) created does" +
                                f" not match population size ({self.evo_config['pop_size']})")

    def set_params(self, params):
        self.params = params

    def set_grads(self, grads):
        self.grads = grads

    def set_fitnesses(self, fitnesses):
        self.fitnesses = np.array(fitnesses)
    
    def select_parents(self):
        argsorted = np.argsort(-self.fitnesses)
        self.parent_params = []
        self.parent_grads = []
        for pop_place in range(len(self.num_mutate)):
            pop_idx = argsorted[pop_place]
            self.parent_params.append(copy.deepcopy(self.params[pop_idx]))
            self.parent_grads.append(copy.deepcopy(self.grads[pop_idx]))
    
    def select_mate_parents(self):
        argsorted = np.argsort(-self.fitnesses)
        self.m_parent_params = []
        self.m_parent_grads = []
        for pop_place in range(5):
            pop_idx = argsorted[pop_place]
            self.m_parent_params.append(copy.deepcopy(self.params[pop_idx]))
            self.m_parent_grads.append(copy.deepcopy(self.grads[pop_idx]))
    
    def create_new_pop(self):
        self.select_parents()
        next_gen = []
        if self.evo_config['hold_elite']:
            next_gen.append(self.parent_params[0])

        for parent_idx in range(len(self.num_mutate)):
            parent_count = self.num_mutate[parent_idx]
            for child_count in range(parent_count):
                child = []
                params = self.parent_params[parent_idx]
                grads = self.parent_grads[parent_idx]
                for i in range(len(params)):
                    child.append(self.mutate(params[i], grads[i]))
                next_gen.append(child)


            #TODO: elegently handle mating option
            """
            self.select_mate_parents()
            parent_combs = combinations([0, 1, 2, 3], 2)
            for mate_count in range(self.evo_config['mate_num']):
                parents = next(parent_combs)
                next_gen.append(self.mate_avg(parents[0], parents[1]))
            """
        self.params = next_gen
        return next_gen
    

    def mutate(self, param, grad):
        learning_rate = random.uniform(self.learning_rate[0], self.learning_rate[1])
        adjusted_grad = learning_rate * grad

        if self.evo_config['mutation_type'] == "gauss":
            locs = param - adjusted_grad
            scales = torch.abs(adjusted_grad) * self.mut_scale

            norm_dist = torch.distributions.normal.Normal(locs, scales)
            return norm_dist.sample()

        elif self.evo_config['mutation_type'] == "uniform":
            locs = param - adjusted_grad
            mutation_amount = torch.abs(adjusted_grad) * self.mut_scale
            dist = torch.distributions.uniform.Uniform(-mutation_amount, mutation_amount)
            return locs + dist.sample()

    def mate_avg(self, parent_1_idx, parent_2_idx):
        learning_rate = random.uniform(self.learning_rate[0], self.learning_rate[1])
        child = []
        for param_1, grad_1, param_2, grad_2 in zip(self.m_parent_params[parent_1_idx], self.m_parent_grads[parent_1_idx],
                                                    self.m_parent_params[parent_2_idx], self.m_parent_grads[parent_2_idx]):
            
            params_1 = param_1 - (grad_1 * learning_rate)
            params_2 = param_2 - (grad_2 * learning_rate)

            child.append(((params_1 + params_2)/2))
        
        return child

    def mate_mask(self, parent_1_idx, parent_2_idx):
        learning_rate = random.uniform(self.learning_rate[0], self.learning_rate[1])
        child = []
        for param_1, grad_1, param_2, grad_2 in zip(self.m_parent_params[parent_1_idx], self.m_parent_grads[parent_1_idx],
                                                    self.m_parent_params[parent_2_idx], self.m_parent_grads[parent_2_idx]):
                        
            rand_mask = torch.randint(0, 2, size=param_1.size())
            inverse = torch.ones_like(rand_mask)

            inverse = inverse - rand_mask
            
            params_1 = param_1 - (grad_1 * learning_rate)
            params_2 = param_2 - (grad_2 * learning_rate)

            new_params_1 = params_1 * rand_mask
            new_params_2 = params_2 * inverse

            child.append(((params_1 + params_2)/2))
        
        return child
    

    def decary_lr(self):
        self.learning_rate = self.lr_decay * self.learning_rate