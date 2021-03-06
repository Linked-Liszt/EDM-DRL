import copy
import torch
import random
import numpy as np
from itertools import combinations


class EA(object):
  def __init__(self, config):
    """
    This class handles the evolutionary aspects of the hybrid algorithm.
    It communicates with the model class through weight and gradient data.

    params:
      config: The experiment config
    """
    self.evo_config = config['earl']
    self.net_config = config['neural_net']

    # CONSTANTS
    self.num_mutate = self.evo_config['recomb_nums'] # [4,3,2,1]
    self.lr = self.evo_config['lr'] #1e-3
    self.lr_decay = self.evo_config['lr_decay']
    self.mut_scale = self.evo_config['mut_scale'] #0.5

    if 'mutation_type' not in self.evo_config:
      self.evo_config['mutation_type'] = 'gauss'
    num_children = 0
    if self.evo_config['hold_elite']:
      num_children += 1
    num_children += sum(self.num_mutate)
    if "mate_num" in self.evo_config:
      num_children += self.evo_config["mate_num"]
    if num_children != self.evo_config['pop_size']:
      raise RuntimeError(
          f"Number children ({num_children}) created does not match population "
          f"size ({self.evo_config['pop_size']})")

  def set_params(self, params):
    self.params = params

  def set_grads(self, grads):
    self.grads = grads

  def set_fitnesses(self, fitnesses):
    self.fitnesses = np.array(fitnesses)

  def _select_parents(self):
    """
    Create a list of parents based of the fitnesses of
    the population members.
    """
    argsorted = np.argsort(-self.fitnesses)
    self.parent_params = []
    self.parent_grads = []
    for pop_place in range(len(self.num_mutate)):
      pop_idx = argsorted[pop_place]
      self.parent_params.append(copy.deepcopy(self.params[pop_idx]))
      self.parent_grads.append(copy.deepcopy(self.grads[pop_idx]))

  def _select_mate_parents(self):
    argsorted = np.argsort(-self.fitnesses)
    self.m_parent_params = []
    self.m_parent_grads = []
    for pop_place in range(5):
      pop_idx = argsorted[pop_place]
      self.m_parent_params.append(copy.deepcopy(self.params[pop_idx]))
      self.m_parent_grads.append(copy.deepcopy(self.grads[pop_idx]))

  def create_new_pop(self):
    """
    Mate and mutate the next generation of members.
    Calls the various functions to select and perform mutation and mating.

    Returns: the parameters of the next generation.
    """
    self._select_parents()
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
          child.append(self._mutate(params[i], grads[i]))
        next_gen.append(child)
      if "mate_num" in self.evo_config:
        self._select_mate_parents()
        parent_combs = combinations([0, 1, 2, 3], 2)
        for mate_count in range(self.evo_config['mate_num']):
          parents = next(parent_combs)
          next_gen.append(self._mate_avg(parents[0], parents[1]))
    self.params = next_gen
    return next_gen

  def _mutate(self, param, grad):
    lr = random.uniform(self.lr[0], self.lr[1])
    adjusted_grad = lr * grad
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

  def _mate_avg(self, parent_1_idx, parent_2_idx):
    lr = random.uniform(self.lr[0], self.lr[1])
    child = []
    for param_1, grad_1, param_2, grad_2 in zip(self.m_parent_params[parent_1_idx],
                                                self.m_parent_grads[parent_1_idx],
                                                self.m_parent_params[parent_2_idx],
                                                self.m_parent_grads[parent_2_idx]):
      params_1 = param_1 - (grad_1 * lr)
      params_2 = param_2 - (grad_2 * lr)
      child.append(((params_1 + params_2)/2))
    return child

  def _mate_mask(self, parent_1_idx, parent_2_idx):
    lr = random.uniform(self.lr[0], self.lr[1])
    child = []
    for param_1, grad_1, param_2, grad_2 in zip(self.m_parent_params[parent_1_idx],
                                                self.m_parent_grads[parent_1_idx],
                                                self.m_parent_params[parent_2_idx],
                                                self.m_parent_grads[parent_2_idx]):
      rand_mask = torch.randint(0, 2, size=param_1.size())
      inverse = torch.ones_like(rand_mask)
      inverse = inverse - rand_mask
      params_1 = param_1 - (grad_1 * lr)
      params_2 = param_2 - (grad_2 * lr)
      new_params_1 = params_1 * rand_mask
      new_params_2 = params_2 * inverse
      child.append(((params_1 + params_2)/2))
    return child

  def decay_lr(self):
    self.lr = self.lr_decay * self.lr
