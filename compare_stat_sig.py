import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import argparse
import os
import scipy.stats as sps
from itertools import permutations

def load_data_v0(log_dict):
    print("Loading log...")
    num_runs = len(nn_dict['experiment_log'])
    len_runs = 500

    gens = []
    best = []
    means = []
    medians = []
    pop_stds = []
    timesteps = []


    for run_idx, run_log in enumerate(nn_dict['experiment_log']):
        gens_run = []
        best_run = []
        means_run = []
        medians_run = []
        pop_stds_run = []
        timesteps_run = []
        if 'test_fit' not in run_log[0]:
            print("WARNING: Test fit not found!")
        for gen_idx, data_dict in enumerate(run_log):
            gens_run.append(data_dict['gen'])
            if 'test_fit' in data_dict:
                fit_val = 'test_fit'
            else:
                fit_val = 'fit_best'
            best_run.append(data_dict[fit_val])
            means_run.append(data_dict['fit_mean'])
            medians_run.append(data_dict['fit_med'])
            pop_stds_run.append(data_dict['fit_std'])
            timesteps_run.append(data_dict['timesteps'])
        
        gens.append(gens_run)
        best.append(best_run)
        means.append(means_run)
        medians.append(medians_run)
        pop_stds.append(pop_stds_run)
        timesteps.append(timesteps_run)

    print("Log Loaded.")
    return timesteps, best, means, medians, pop_stds, stds, gens


def interp_load_data_v0(log_dict):
    print("Loading log...")
    num_runs = len(nn_dict['experiment_log'])

    gens = []
    best = []
    means = []
    medians = []
    pop_stds = []
    timesteps = []


    for run_idx, run_log in enumerate(nn_dict['experiment_log']):
        # I'm sorry, this is very ugly
        gens_run = []
        best_run = []
        means_run = []
        medians_run = []
        pop_stds_run = []
        timesteps_run = []
        for gen_idx, data_dict in enumerate(run_log):
            gens_run.append(gen_idx)
            if 'test_fit' in data_dict:
                fit_val = 'test_fit'
            else:
                fit_val = 'fit_best'
                print("WARNING: Test fit not found!")
            best_run.append(data_dict[fit_val])
            means_run.append(data_dict['fit_mean'])
            medians_run.append(data_dict['fit_med'])
            pop_stds_run.append(data_dict['fit_std'])
            timesteps_run.append(data_dict['timesteps'])
        
        interp_points = np.arange(0, nn_dict['config']['experiment']['timesteps'] + 1, step=1000)
    
        best_interp = np.interp(interp_points, timesteps_run, best_run, left=0.0, right=best_run[-1])

        best.append(best_interp)

    std = np.std(best, axis=0)
    best = np.mean(best, axis=0)
    
    print("Log Loaded.")
    return interp_points, best, std




def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_paths', metavar='log files or folders', type=str, nargs='+',
                         help='log files or folder to scan')

    parser.add_argument('--f', action='store_true', dest='folder_flag',
                         help='scan entire folder of logs')

    parser.add_argument('--i', action='store_true', dest='ignore_failed',
                         help='ignore runs that didn\'t solve the task')

    return parser.parse_args()

def scan_folder(folder_paths):
    log_files = []
    for folder_path in folder_paths:
        for file in os.listdir(folder_path):
            if file.endswith('.p'):
                log_files.append(os.path.join(folder_path, file))
    return log_files

def get_log_name(path):
    log_name = path
    if '\\' in path:
        log_name = path[path.rfind('\\') + 1:path.rfind('.')]
    elif '/' in path:
        log_name = path[path.rfind('/') + 1:path.rfind('.')]
    return log_name



if __name__ == '__main__':
    parser = parse_arguments()

    if parser.folder_flag:
        print(parser.log_paths)
        log_paths = scan_folder(parser.log_paths)
    else:
        log_paths = parser.log_paths

    plotted_log_paths = []
    timesteps = []
    bests = []
    means = []
    medians = []
    pop_stds = []
    stds = []
    gens = []
    
    interp_stds = []
    for path in log_paths:
        nn_dict = pickle.load(open(path, 'rb'))
    
        timestep, best, mean, median, pop_std, std, gen = load_data_v0(nn_dict)
        _, _, interp_std = interp_load_data_v0(nn_dict)


        if parser.ignore_failed and best[-1] < 50:
            continue

        plotted_log_paths.append(path)
        timesteps.append(timestep)
        bests.append(best)
        means.append(mean)
        medians.append(median)
        pop_stds.append(pop_std)
        stds.append(std)
        interp_stds.append(interp_std)
        gens.append(gen)



    # Calculate the average and variance of solve time. 
    solve_score = 475.0
    solve_times = []
    solve_generations = []
    did_solve = 0

    for path_idx, path in enumerate(plotted_log_paths):
        solves = []
        for run_idx, runs in enumerate(bests[path_idx]):
            for gen_idx, best, timestep in zip(gens[path_idx][run_idx], bests[path_idx][run_idx], timesteps[path_idx][run_idx]):
                if best >= solve_score:
                    solves.append(timestep)
                    solve_generations.append(gen_idx)
                    did_solve += 1
                    break
        solve_times.append(solves)


for log_idxs in permutations(range(0, len(plotted_log_paths)), 2):

    print(plotted_log_paths[log_idxs[0]])
    print(plotted_log_paths[log_idxs[1]])

    var_0_list = solve_times[log_idxs[0]]
    var_1_list = solve_times[log_idxs[1]]

    print(f"{len(var_0_list)} samples loaded into variable 0")
    print(f"{len(var_1_list)} samples loaded into variable 1")
    print("------------------------------------")

    # Begin F Test
    print("Beginning F Test")
    print("------------------------------------")
    mean_var_0 = np.mean(var_0_list)
    mean_var_1 = np.mean(var_1_list)
    print(f"mean var 0: {mean_var_0}")
    print(f"mean var 1: {mean_var_1}")
    f = np.var(var_0_list)/np.var(var_1_list)
    print(f"F: {f}")
    f_crit = sps.f.cdf(f, len(var_0_list) - 1, len(var_1_list) - 1)
    print(f"F critical: {f_crit}")
    print("------------------------------------")

    # Are variances equal?
    print("Are variances equal?")
    print("------------------------------------")
    print("is mean(var 0) > mean(var 2) and F < F critical?")
    print("or is mean(var 0) < mean(var 2) and F > F critical? ")
    print("------------------------------------")
    var_equal = (mean_var_0 > mean_var_1) and (f < f_crit)
    var_equal = var_equal or ((mean_var_0 < mean_var_1) and (f > f_crit))
    if var_equal:
        print("Variances are Equal")
    else:
        print("Variances are Unequal")
    print("------------------------------------")

    if var_equal:
        print("Begin two-tailed two-sample t-test assuming equal variances")
    else:
        print("Begin two-tailed two-sample t-test assuming unequal variances")
    t_t_test, p_t_test = sps.ttest_ind(var_0_list, var_1_list, equal_var=var_equal)
    print(f"T-Test Value: {t_t_test}")
    print(f"Two tailed p: {p_t_test}")
    print("------------------------------------")
    print("Assuming alpha=0.05")
    print("------------------------------------")
    if p_t_test > 0.05:
        print("Null Hypothesis Accepted, Algorithm is NOT statistically better")
    else:
        print("Null Hypothesis Rejected, Algorithm is statistically better")

    input()
