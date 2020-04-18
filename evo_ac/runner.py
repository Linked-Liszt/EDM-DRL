import gym
import numpy as np
from evo_ac.model import EvoACModel
import torch
from evo_ac.storage import EvoACStorage
from evo_ac.grad_evo import EvoACEvoAlg
from evo_ac.logger import EvoACLogger
import scipy.special as sps

class EvoACRunner(object):
    def __init__(self, config):
        self.config = config
        self.config_evo = config['evo_ac']
        self.config_net = config['neural_net']
        self.config_exp = config['experiment']

        if self.config_exp['env'] == "CartPole-v1":
            self.stop_fit = 475.0
        elif self.config_exp['env'] == "LunarLander-v2":
            self.stop_fit = 200.0
            

        self.env = gym.make(self.config_exp['env'])
        self.test_env = gym.make(self.config_exp['env'])
        self.logger = EvoACLogger(config)

        
    def train(self):
        for run_idx in range(self.config_exp['num_runs']):
            self.reset_experiment()
            self.timesteps = 0
            self.stop_counter = 0
            self.last_log = -9999999

            for self.gen_idx in range(10000):
                self.storage.reset_storage()

                for pop_idx in range(self.config_evo['pop_size']):
                    obs = self.env.reset()

                    fitness = 0

                    while True:

                        action, log_p_a, entropy, value = self.model.get_action(self.storage.obs2tensor(obs), pop_idx)

                        self.timesteps += 1

                        obs, reward, done, info = self.env.step(action.cpu().numpy())
                        fitness += reward

                        self.storage.insert(pop_idx, reward, action, log_p_a, value, entropy)

                    
                        if done:
                            break
                    
                    self.storage.insert_fitness(pop_idx, fitness)
                
                self.update_evo_ac()

                if self.timesteps - self.last_log >= self.config_exp['log_interval'] or self.timesteps > self.config_exp['timesteps']:
                    test_fit = self.test_algorithm()

                    self.logger.save_fitnesses(self.model, test_fit, self.storage.fitnesses, self.policy_loss_log, 
                                                self.value_loss_log, self.gen_idx, self.timesteps)
                    self.logger.print_data(self.gen_idx)
                    self.last_log = self.timesteps

                    if test_fit >= self.stop_fit:
                        break

                self.model.insert_params(self.new_pop)


                if self.timesteps > self.config_exp['timesteps']:
                    break

            self.logger.end_run()
        self.logger.end_experiment()


    def update_evo_ac(self):
        self.model.opt.zero_grad()
        loss, self.policy_loss_log, self.value_loss_log = self.storage.get_loss()
        loss.backward()
        self.evo.set_grads(self.model.extract_grads())
        
        self.model.opt.step()

        self.evo.set_fitnesses(self.storage.fitnesses)

        with torch.no_grad():
            self.new_pop = self.evo.create_new_pop()

    def reset_experiment(self):
        obs_size = np.prod(np.shape(self.env.observation_space))
        num_pop = self.config_evo['pop_size']
        max_ep_steps = self.env._max_episode_steps
        value_coeff = self.config_evo['value_coeff']
        entropy_coff = self.config_evo['entropy_coeff']

        print("NEW RUN")

        self.storage = EvoACStorage(num_pop, self.config)
        self.model = EvoACModel(self.config)
        self.evo = EvoACEvoAlg(self.config)
        self.evo.set_params(self.model.extract_params())

    def test_algorithm(self):
        with torch.no_grad():
            fitnesses = []

            for _ in range(100):
                fitness = 0
                obs = self.test_env.reset()
                while True:
                    action = self.get_test_action(obs)
                    obs, rewards, done, info = self.test_env.step(action)
                    fitness += rewards
                    if done:
                        break
                fitnesses.append(fitness)
            return np.mean(fitnesses)

    def get_test_action(self, obs):
        obs = self.storage.obs2tensor(obs)
        fitnesses = self.storage.fitnesses
        if self.config_exp['test_strat'] == 'best':
            best_pop = np.argmax(fitnesses)
            action, _, _, _ = self.model.get_action(obs, best_pop)
            action = action.cpu().numpy()
        elif self.config_exp['test_strat'] == 'softmax':
            probs = sps.softmax(fitnesses)
            pop_idx = np.random.choice(self.config_evo['pop_size'], 1, p=probs)
            action, _, _, _ = self.model.get_action(obs, pop_idx[0])
            action = action.cpu().numpy()
        elif self.config_exp['test_strat'] == 'weightedvote':
            actions = [self.model.get_action(obs, pop_idx)[0].item() for pop_idx in range(self.config_evo['pop_size'])]
            action_votes = [0] * self.test_env.action_space.n
            for mod_action, weight in zip(actions, fitnesses):
                action_votes[mod_action] += weight
            action = np.argmax(action_votes)
        return action