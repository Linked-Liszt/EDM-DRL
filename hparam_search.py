import numpy as np
import gym
import json
import sys
from evo_ac.runner import EvoACRunner
import matplotlib.pyplot as plt
import random


# Search params adaped from https://arxiv.org/pdf/1912.02877.pdf

if __name__ == '__main__':
    
    config_path = sys.argv[1]
    with open(config_path, 'r') as config_file:
        config_dict = json.load(config_file)

    


    for hparam_run_idx in range(300):
        config_dict['experiment']['num_runs'] = 7
        config_dict['experiment']['log_path'] = "/home/oxymoren/Desktop/EA/ea-safety/checkpoints/hparam_search_ll"
        config_dict['experiment']['log_name'] = f"evo_ac_ll_hparam_search_1_{hparam_run_idx}"
        input_size = 8
        output_size = 4


        activation_function = random.choice(['ReLU', 'Tanh'])
        
        config_dict['experiment']['gamma'] = random.choice([0.98, 0.99, 0.995, 0.999])
        config_dict['evo_ac']['entropy_coeff'] = random.choice([0.0, 0.01, 0.02, 0.05, 0.1])
        config_dict['evo_ac']['value_coeff'] = random.choice([0.1, 0.2, 0.5, 1.0])
        config_dict['evo_ac']['lr_decay'] = random.choice([0.95, 0.97, 0.99])

        lr = random.choice(np.logspace(-6, -2,num=50))
        config_dict['evo_ac']['learning_rate'] = [lr,lr]
        lr_2 = random.choice(np.logspace(-6, -2,num=50))
        config_dict['neural_net']['learning_rate'] = lr_2


        arch = random.randint(0, 3)
        size_set = random.choice([[32, 64], [64, 128]])
        shared = policy = value = None
        if arch == 0:
            shared = [
                {
                    "type": "Linear",
                    "params": [input_size, size_set[0]], 
                    "kwargs": {"bias":True}
                }, 
                {
                    "type": activation_function,
                    "params": [], 
                    "kwargs": {}
                }
                ]
            policy = [
                {
                    "type": "Linear",
                    "params": [size_set[0], output_size], 
                    "kwargs": {"bias":True}
                }
                ]
            value = [
                {
                    "type": "Linear",
                    "params": [size_set[0], 1], 
                    "kwargs": {"bias":True}
                }
                ]

        elif arch == 1:
            shared = [
                {
                    "type": "Linear",
                    "params": [input_size, size_set[0]], 
                    "kwargs": {"bias":True}
                }, 
                {
                    "type": activation_function,
                    "params": [], 
                    "kwargs": {}
                },
                {
                    "type": "Linear",
                    "params": [size_set[0], size_set[1]], 
                    "kwargs": {"bias":True}
                }, 
                {
                    "type": activation_function,
                    "params": [], 
                    "kwargs": {}
                }
                ]
            policy = [
                {
                    "type": "Linear",
                    "params": [size_set[1], output_size], 
                    "kwargs": {"bias":True}
                }
                ]
            value = [
                {
                    "type": "Linear",
                    "params": [size_set[1], 1], 
                    "kwargs": {"bias":True}
                }
                ]

        elif arch == 2:
            shared = [
                {
                    "type": "Linear",
                    "params": [input_size, size_set[0]], 
                    "kwargs": {"bias":True}
                }, 
                {
                    "type": activation_function,
                    "params": [], 
                    "kwargs": {}
                }
                ]
            policy = [
                {
                    "type": "Linear",
                    "params": [size_set[0], size_set[1]], 
                    "kwargs": {"bias":True}
                },
                {
                    "type": activation_function,
                    "params": [], 
                    "kwargs": {}
                },
                {
                    "type": "Linear",
                    "params": [size_set[1], output_size], 
                    "kwargs": {"bias":True}
                }
                ]
            value = [
                {
                    "type": "Linear",
                    "params": [size_set[0], size_set[1]], 
                    "kwargs": {"bias":True}
                }, 
                {
                    "type": activation_function,
                    "params": [], 
                    "kwargs": {}
                },
                {
                    "type": "Linear",
                    "params": [size_set[1], 1], 
                    "kwargs": {"bias":True}
                }
                ]

        elif arch == 3:
            shared = [
                {
                    "type": "Linear",
                    "params": [input_size, size_set[0]], 
                    "kwargs": {"bias":True}
                }, 
                {
                    "type": activation_function,
                    "params": [], 
                    "kwargs": {}
                },
                {
                    "type": "Linear",
                    "params": [size_set[0], size_set[1]], 
                    "kwargs": {"bias":True}
                }, 
                {
                    "type": activation_function,
                    "params": [], 
                    "kwargs": {}
                },

                ]
            policy = [
                {
                    "type": "Linear",
                    "params": [size_set[1], size_set[1]], 
                    "kwargs": {"bias":True}
                },
                {
                    "type": activation_function,
                    "params": [], 
                    "kwargs": {}
                },
                {
                    "type": "Linear",
                    "params": [size_set[1], output_size], 
                    "kwargs": {"bias":True}
                }
                ]
            value = [
                {
                    "type": "Linear",
                    "params": [size_set[1], size_set[1]], 
                    "kwargs": {"bias":True}
                }, 
                {
                    "type": activation_function,
                    "params": [], 
                    "kwargs": {}
                },
                {
                    "type": "Linear",
                    "params": [size_set[1], 1], 
                    "kwargs": {"bias":True}
                }
            ]

        config_dict['neural_net']['shared'] = shared
        config_dict['neural_net']['policy'] = policy
        config_dict['neural_net']['value'] = value
        
        
        runner = EvoACRunner(config_dict)
        runner.train()