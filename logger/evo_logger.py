import pickle
import numpy as np
import copy
import os
from datetime import datetime
import json
import numpy as np

class EvoLogger(object):
    """
    This class helps maintain a log of the experiment. 
    Other data types can be attached for enhanced logging features. 
    All objects MUST be pickalbe. 

    Args:
        log_name: the name of the logs to create. 

        output folder: the filepath to output the log files to. 

        experiment config: I use json-style dicts to save configs. 
            If you use something similar, you're welcome to attach it here. 

        log_run:
            if enabled, creates an intermediate log at the end of every run
            recommended to leave off unless the run is expected to end prematurely
    """
    def __init__(self, log_name, output_folder=None, experiment_config=None, log_run=False, use_pickle=True):
        

        self.output_folder = output_folder
        self.log_name = log_name
        self.use_pickle = use_pickle

        self.experiment_config = experiment_config
        
        self.experiment_log = []
        self.run_log = []
        self.run_counter = 0
        
        self.start_time = datetime.now()
        self.run_end_times = []
        self.run_end_times.append(self.start_time)

        self.log_run = log_run
        self.run_counter = 0

        if output_folder is not None:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)


    def save_sample(self, test_fit, timesteps, gen, model=None, pop_fitnesses=None, loss=None, diversity=None):
        """
        Save a sample of the algorithm. Feel free to add more parameters and things to save as needed. 

        Args:
            test_fit[float]: the result of the 100 run test of the algorithm. (average of 100 runs) 
            (does not count towards timesteps)

            timesteps[int]: the number of timesteps spent training before the test was conducted

            gen[int]: the generation at which the test was conducted. 

            model [optional]: if the model is pickalable, it can be saved here. 

            pop_fitnesses [optional][list or np array]: the fitness of each individual at time of experiment as an 

            loss [optional][float]: the loss of network at time of testing. 

            diversity[optional]: any diversity measure
        """

        data_dict = {}
        data_dict['test_fit'] = test_fit
        data_dict['timesteps'] = timesteps
        data_dict['gen'] = gen
        data_dict['model'] = model
        data_dict['pop_fitnesses'] = pop_fitnesses
        data_dict['loss'] = loss
        data_dict['diversity'] = diversity
        self.run_log.append(data_dict)


    def end_run(self):
        """
        Ends the run and resets internal variables. 
        If log_run is enabled, will export intermediate logs. 
        """
        self.experiment_log.append(self.run_log)
        self.run_log = []
        self.run_end_times.append(datetime.now())
        if self.log_run:
            self._export_data_json_backup(f'run_{self.run_counter:02d}')
            if self.use_pickle:
                self._export_data(f'run_{self.run_counter:02d}')
        self.run_counter += 1

    def end_experiment(self):
        """
        Ends the experiment and logs the final log. 
        """
        self._export_data_json_backup('final')
        if self.use_pickle:
            self._export_data('final')
    

    def _build_path(self):
        if self.output_folder == None:
            return self.log_name + '_'
        else:
            return self.output_folder + '/' + self.log_name + '_'
        
    def _export_data(self, export_name):
        """
        Internal function to handle exporting of data. 
        If another file of the same name exists, appends the datetime to the file. 
        
        Args:
            export_name: the name of the files to export the data as. 
        """
        data_path = self._build_path() + export_name
         
        if os.path.isfile(data_path):
            data_path += datetime.now().strftime("%d_%H_%M_%S")

        data_path += '.p'

        save_dict = {}
        save_dict['start_time'] = self.start_time
        save_dict['experiment_log'] = self.experiment_log
        save_dict['times'] = self.run_end_times
        save_dict['config'] = self.experiment_config
        save_dict['version'] = 'v1'
        save_dict['log_name'] = self.log_name
        pickle.dump(save_dict, open(data_path, 'wb'))

    def _export_data_json_backup(self, export_name):
        """
        Internal function to handle exporting of data.
        This one exports a json backup in case the pickle fails.  

        If another file of the same name exists, appends the datetime to the file. 
        
        Args:
            export_name: the name of the files to export the data as. 
        """
        data_path = self._build_path() + export_name
         
        if os.path.isfile(data_path):
            data_path += datetime.now().strftime("%d_%H_%M_%S")

        data_path += '.json'
        
        exp_data = []
        for run_dict in self.experiment_log:
            exp_run_dict = {}
            test_fits = []
            timesteps = []
            gens = []
            for data_dict in run_dict:
                test_fits.append(data_dict['test_fit'])
                timesteps.append(data_dict['timesteps'])
                gens.append(data_dict['gen'])
            exp_run_dict['test_fit'] = test_fits
            exp_run_dict['timesteps'] = timesteps
            exp_run_dict['gens'] = gens
            exp_data.append(exp_run_dict)

        save_dict = {}
        save_dict['version'] = 'v1_json'
        save_dict['log_name'] = self.log_name
        save_dict['experiment_data'] = exp_data
        with open(data_path, "w") as json_file:
            json_file.write(json.dumps(save_dict, indent=4, sort_keys=True))

    def print_data(self):
        """
        Prints log statistics to the console.
        """
        data_dict = self.run_log[-1]
        display_str = f"\nRun {self.run_counter}  |  Gen {data_dict['gen']}  |  Timesteps {data_dict['timesteps']} \n" \
            + f"Test Fitness: {data_dict['test_fit']}\n"
        
        if data_dict['pop_fitnesses'] is not None:
            display_str += f"Population Best: {max(data_dict['pop_fitnesses'])}  |  Population Mean: {np.mean(data_dict['pop_fitnesses'])}" 
            display_str += f"  |  Population Var: {np.std(data_dict['pop_fitnesses']):.2f}\n" 
        
        if data_dict['loss'] is not None:
            display_str += f"Loss: {data_dict['loss']:.2e}\n"
        
        display_str += f"Experiment: {self.log_name}\n"
        print(display_str)

