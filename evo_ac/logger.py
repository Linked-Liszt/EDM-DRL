import pickle
import numpy as np
import copy
import os
from datetime import datetime

NOTIFICATION_PATH = "/home/oxymoren/Desktop/rand_utils/dm_me.sh"

class EvoACLogger(object):
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
        self.experiment_log.append(self.run_log)
        self.run_log = []
        self.run_end_times.append(datetime.now())
        if self.config_exp['log_run']:
            self._export_data(f'run_{self.run_counter:02d}')
        self.run_counter += 1

    def end_experiment(self):
        self._export_data('final')
        if 'discord_ping' in self.config_exp and self.config_exp['discord_ping']:
            self._send_discord_notification()
        
    def _export_data(self, export_name):
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

    def print_data(self, gen_idx):
        if gen_idx % self.print_interval == 0:
            data_dict = self.run_log[-1]
            display_str = f"\n\nRun {self.run_counter}  Gen {gen_idx}  Timesteps {data_dict['timesteps']} \n" \
                + f"Best: {data_dict['fit_best']}  Mean: {data_dict['fit_mean']}  Test: {data_dict['test_fit']}\n" \
                + f"Policy Loss: {data_dict['policy_loss']:.2e}  Value Loss: {data_dict['value_loss']:.2e}\n" \
                + f"Full: {data_dict['fit']}\n"\
                + f"Experiment: {self.config_exp['log_name']}"
            print(display_str)
            
    def _send_discord_notification(self):
        end_time = datetime.now()
        time_delta = end_time - self.start_time

        time_start_str = self.start_time.strftime("%H:%M:%S")
        end_time_str = end_time.strftime("%H:%M:%S")
        time_delta_str = str(time_delta)
        command = NOTIFICATION_PATH + \
            f' \"Experiment {self.name.rstrip("_")} has finished.' + \
            f'//n// Time Start: {time_start_str} Time End: {end_time_str}' + \
            f'//n// Time Δ: {time_delta_str}\"'
        os.system(command)

