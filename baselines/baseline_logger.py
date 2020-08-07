import pickle
import numpy as np
import copy
import os
from datetime import datetime


class BaseLineLogger(object):
    """
    Handles experiment data logging and creation of logging artifacts.

    Params:
        config (dict): the experiment config

    """
    def __init__(self, name, env, directory):
        self.env = env
        self.name = name
        self.directory = directory

        self.experiment_log = []
        self.run_log = []
        self.run_counter = 0

        self.start_time = datetime.now()
        self.run_end_times = []
        self.run_end_times.append(self.start_time)

        if not os.path.exists(self.directory):
            try:
                os.makedirs(self.directory)
            except OSError:
                print("WARNING: Unable to create log directory")


    def save_fitnesses(self, test_fit, timesteps, gen):
        """
        Saves the fitness of a test run. Also takes some
        loss metrics of the last training step.

        """
        data_dict = {}
        data_dict['gen'] = gen
        data_dict['timesteps'] = timesteps
        data_dict['test_fit'] = test_fit
        data_dict['fit'] = [5,5,5,5,5]
        data_dict['fit_best'] = 5
        data_dict['fit_mean'] = 5
        data_dict['fit_med'] = 5
        data_dict['fit_std'] = 5
        data_dict['policy_loss'] = 0.0
        data_dict['value_loss'] = 0.0
        self.run_log.append(data_dict)

    def end_run(self):
        """
        Resets internal variables to prepare for a new run.
        Saves previous run data.
        """
        self.experiment_log.append(self.run_log)
        self.run_log = []
        self.run_end_times.append(datetime.now())
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
        data_path = os.path.join(self.directory, (self.name + '_' + export_name + '.p'))

        save_dict = {}
        save_dict['start_time'] = self.start_time
        save_dict['end_time'] = datetime.now()
        save_dict['env'] = self.env
        save_dict['experiment_log'] = self.experiment_log
        save_dict['times'] = self.run_end_times
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
        display_str = f"\n\nRun {self.run_counter}  Timesteps {data_dict['timesteps']} \n" \
            + f"Test: {data_dict['test_fit']}\n"
        print(display_str)
