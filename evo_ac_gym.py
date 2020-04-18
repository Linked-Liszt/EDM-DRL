import numpy as np
import gym
import json
import sys
from evo_ac.runner import EvoACRunner
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    config_path = sys.argv[1]
    with open(config_path, 'r') as config_file:
        config_dict = json.load(config_file)

    runner = EvoACRunner(config_dict)
    runner.train()