import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import argparse
import os


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
            timesteps_run.append(data_dict['timesteps'])
        
        interp_points = np.arange(0, 100000 + 1, step=1000)
    
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
    
        timestep, best, _, _, _, _, gen = load_data_v0(nn_dict)
        _, _, interp_std = interp_load_data_v0(nn_dict)


        if parser.ignore_failed and best[-1] < 50:
            continue

        plotted_log_paths.append(path)
        timesteps.append(timestep)
        bests.append(best)
        interp_stds.append(interp_std)
        gens.append(gen)



    # Calculate the average and variance of solve time. 
    if not parser.folder_flag:
        solve_score = 475.0
        solve_times = []
        solve_generations = []
        did_solve = 0

        for path_idx, path in enumerate(plotted_log_paths):
            solves = []
            for run_idx, runs in enumerate(bests[path_idx]):
                for gen_idx, best, timestep in zip(gens[path_idx][run_idx], bests[path_idx][run_idx], timesteps[path_idx][run_idx]):
                    if best >= solve_score:
                        solves.append(int(timestep))
                        solve_generations.append(gen_idx)
                        did_solve += 1
                        break
            solve_times.append(solves)

        exp_stds = interp_stds[0]
        print(exp_stds[-1])
        stds_no_zero = exp_stds[exp_stds > 3.0]

        print(plotted_log_paths[0])    
        print(f"Num Runs: {len(bests[0])}")
        print(f"Num Solves: {did_solve}")
        print(f"Gens Till Solve {np.mean(solve_generations)}")
        print(f"STD Gens Till Solve {np.std(solve_generations)}")
        print(f"Average Solve Timesteps: {np.mean(solve_times)}" )
        print(f"Standard Deviation of Solve Timesteps: {np.std(solve_times)}")
        print(f"Mean of STDs: {np.mean(stds_no_zero)}")



        # Graph data
        fig, ax_h = plt.subplots(figsize=(13,7), sharex=True)
        ax_h.set_ylabel("Fitness (Mean of 100 runs)", fontsize=15)
        ax_h.set_xlabel("Timesteps", fontsize=15)


    for path_idx, path in enumerate(plotted_log_paths):
        for run_idx, runs in enumerate(bests[path_idx]):
            run_name = f"Run: run_idx {run_idx}"

            ax_h.plot(timesteps[path_idx][run_idx], bests[path_idx][run_idx], label=run_name)

    ax_h.legend(loc='upper center', bbox_to_anchor=(1.05, 1.5), shadow=True, fontsize=7)
    #ax_l.legend()
    fig.suptitle("WeightedVote Runs", fontsize=25)

    plt.show()