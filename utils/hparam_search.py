import numpy as np
import gym
import json
import sys
from earl.runner import EvoACRunner
import matplotlib.pyplot as plt
import random
import numpy as np
import optuna
from multiprocessing import Lock


NUM_JOBS = 3
TXT_LOG_PATH = 'log.txt'
NUM_RUNS = 2
LOG_LOCK = Lock()

config_path = sys.argv[1]
with open(config_path, 'r') as config_file:
    CONFIG = json.load(config_file)


def edm_opt(trial: optuna.Trial) -> int:
    global CONFIG
    global NUM_RUNS
    new_config = CONFIG
    new_config['experiment']['num_runs'] = 1
    new_config['experiment']['log_path'] = "/home/oxymoren/Desktop/EA/ea-safety/checkpoints/hparam_search"
    input_size = 8
    output_size = 4

    new_config['experiment']['gamma'] = trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    new_config['earl']['entropy_coeff'] = trial.suggest_loguniform('ent_coef', 0.00000001, 0.1)
    new_config['earl']['value_coeff'] = trial.suggest_categorical('value_coeff', [0.1, 0.2, 0.5, 1.0])
    new_config['earl']['lr_decay'] = trial.suggest_categorical('lr_decay', [0.95, 0.97, 0.99])

    lr_evo = trial.suggest_loguniform('lr_evo', 1e-5, 1)
    new_config['earl']['lr'] = [lr_evo, lr_evo]
    new_config['neural_net']['lr'] = trial.suggest_loguniform('lr_opt', 1e-7, 1)

    timesteps = []
    for _ in range(NUM_RUNS):
        runner = EvoACRunner(new_config)
        runner.train()
        if runner.timesteps >= 100000:
            timesteps.append(100000)

    return np.mean(timesteps)

def save_trials(study: optuna.Study, frozen_trial: optuna.trial.FrozenTrial) -> None:
    global TXT_LOG_PATH
    global LOG_LOCK
    with LOG_LOCK:
        with open(TXT_LOG_PATH, 'a') as log_f:
            log_f.write(str(frozen_trial))
            log_f.write('\n')

if __name__ == '__main__':
    sampler = optuna.samplers.TPESampler(**optuna.samplers.TPESampler.hyperopt_parameters())
    study = optuna.create_study(sampler=sampler, direction='minimize')
    study.optimize(edm_opt, n_trials=200, callbacks=[save_trials], n_jobs=NUM_JOBS)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
