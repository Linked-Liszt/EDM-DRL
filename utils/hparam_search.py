import numpy as np
import gym
import json
import sys
from earl.runner import EvoACRunner
import matplotlib.pyplot as plt
import random
import optuna

config_path = sys.argv[1]
with open(config_path, 'r') as config_file:
    CONFIG = json.load(config_file)

COUNTER = 0

TXT_LOG_PATH = 'log_large.txt'


def edm_opt(trial: optuna.Trial) -> int:
    global CONFIG
    global COUNTER
    CONFIG['experiment']['num_runs'] = 1
    CONFIG['experiment']['log_path'] = "/home/oxymoren/Desktop/EA/ea-safety/checkpoints/hparam_search"
    CONFIG['experiment']['log_name'] = f"earl_hparam_search_1_{COUNTER}"
    input_size = 8
    output_size = 4

    CONFIG['experiment']['gamma'] = trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    CONFIG['earl']['entropy_coeff'] = trial.suggest_loguniform('ent_coef', 0.00000001, 0.1)
    CONFIG['earl']['value_coeff'] = trial.suggest_categorical('value_coeff', [0.1, 0.2, 0.5, 1.0])
    CONFIG['earl']['lr_decay'] = trial.suggest_categorical('lr_decay', [0.95, 0.97, 0.99])

    lr_evo = trial.suggest_loguniform('lr_evo', 1e-5, 1)
    CONFIG['earl']['lr'] = [lr_evo, lr_evo]
    CONFIG['neural_net']['lr'] = trial.suggest_loguniform('lr_opt', 1e-7, 1)


    runner = EvoACRunner(CONFIG)
    runner.train()

    COUNTER += 1
    if runner.timesteps >= 100000:
        return 100000
    return runner.timesteps

def save_trials(study: optuna.Study, frozen_trial: optuna.trial.FrozenTrial) -> None:
    global TXT_LOG_PATH
    with open(TXT_LOG_PATH, 'w') as log_f:
        for trial in study.trials:
            log_f.write(str(trial))
            log_f.write('\n')



if __name__ == '__main__':
    sampler = optuna.samplers.TPESampler(**optuna.samplers.TPESampler.hyperopt_parameters())
    study = optuna.create_study(sampler=sampler, direction='minimize')
    study.optimize(edm_opt, n_trials=200, callbacks=[save_trials])

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
