import pickle
import numpy as np
import copy
import os
from datetime import datetime


class EvoACLogger(object):
    """
    Handles experiment data logging and creation of logging artifacts. 

    Params:
        config (dict): the experiment config

    """
    def __init__(self, config):

        self.config = config
        self.config_exp = config['experiment']
        self.config_evoac = config['evo_ac']

        self.directory = self.config_exp['log_path']
        self.name = self.config_exp['log_name']

        self.print_interval = self.config_exp['print_interval']
        
        self.env = self.config_exp['env']
        
        self.experiment_log = []
        self.run_log = []
        self.run_counter = 0
        self.best_fitness = float('-inf')
        
        self.start_time = datetime.now()
        self.run_end_times = []
        self.run_end_times.append(self.start_time)

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)


    def save_fitnesses(self, model, test_fit, fitnesses, policy_loss, value_loss, gen, timesteps):
        """
        Saves the fitness of a test run. Also takes some 
        loss metrics of the last training step.

        """
        data_dict = {}
        data_dict['gen'] = gen
        data_dict['timesteps'] = timesteps
        data_dict['test_fit'] = test_fit
        data_dict['fit'] = copy.deepcopy(fitnesses)
        data_dict['fit_best'] = np.max(fitnesses)
        data_dict['fit_mean'] = np.mean(fitnesses)
        data_dict['fit_med'] = np.median(fitnesses)
        data_dict['fit_std'] = np.std(fitnesses)
        data_dict['policy_loss'] = policy_loss
        data_dict['value_loss'] = value_loss
        self.run_log.append(data_dict)

        if float(np.max(fitnesses)) > self.best_fitness:
            self.best_model =  copy.deepcopy(model)

    def end_run(self):
        """
        Resets internal variables to prepare for a new run. 
        Saves previous run data. 
        """
        self.experiment_log.append(self.run_log)
        self.run_log = []
        self.run_end_times.append(datetime.now())
        if self.config_exp['log_run']:
            self._export_data(f'run_{self.run_counter:02d}')
        self.run_counter += 1

    def end_experiment(self):
        """
        Steps to take at the end of the experiment. 
        For now, just data exporting. 
        """
        self._export_data('final')
        
    def _export_data(self, export_name):
        """
        Exports data to the log path as specified in the config. 
        Checks for conflits. The data is exported as a pickle.

        export_name (str): name of the log file to create 
        """
        data_path = self.directory + '/' + self.name + '_' \
                + export_name
         
        if os.path.isfile(data_path):
            data_path += datetime.now().strftime("%d_%H_%M_%S")

        data_path += '.p'

        save_dict = {}
        save_dict['start_time'] = self.start_time
        save_dict['env'] = self.env
        save_dict['best_nn'] = self.best_model
        save_dict['experiment_log'] = self.experiment_log
        save_dict['times'] = self.run_end_times
        save_dict['config'] = self.config
        save_dict['version'] = 'v0'
        pickle.dump(save_dict, open(data_path, 'wb'))

    def print_data(self):
        """
        Retrieves data from the latest log object and prints it to console. 
        Gen idx is used to change the interval which data is printed. 

        Params:
            gen_idx (int): used to determine whether to 
        """
        data_dict = self.run_log[-1]
        display_str = f"\n\nRun {self.run_counter}  Gen {data_dict['gen']}  Timesteps {data_dict['timesteps']} \n" \
            + f"Best: {data_dict['fit_best']}  Mean: {data_dict['fit_mean']}  Test: {data_dict['test_fit']}\n" \
            + f"Policy Loss: {data_dict['policy_loss']:.2e}  Value Loss: {data_dict['value_loss']:.2e}\n" \
            + f"Full: {data_dict['fit']}\n"\
            + f"Experiment: {self.config_exp['log_name']}"
        print(display_str)
        