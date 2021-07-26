'''
-------------------------
Written for Python 2.7 and 3.7
Python modules required:
-Numpy
-Scipy
-dadi
-------------------------

Daniel Portik
daniel.portik@gmail.com
https://github.com/dportik
Updated September 2019
'''
import sys
import os
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import dadi
from datetime import datetime as dt
import pandas as pd
import time

from Replicate_Task import Replicate_Task

def parse_params(param_number, in_params=None, in_upper=None, in_lower=None):
    """    
    Function to correctly deal with parameters and bounds, and if none were provided, 
    to generate them automatically.
    
    Arguments
    param_number: number of parameters in the model selected (can count in params line for the model)
    in_params: a list of parameter values 
    in_upper: a list of upper bound values
    in_lower: a list of lower bound values
    """
    param_number = int(param_number)

    # param set
    if in_params is None:
        params = [1] * param_number
    elif len(in_params) != param_number:
        raise ValueError(
            "Set of input parameters does not contain the correct number of values: {}".format(param_number))
    else:
        params = in_params

    # upper bound
    if in_upper is None:
        upper_bound = [30] * param_number
    elif len(in_upper) != param_number:
        raise ValueError(
            "Upper bound set for parameters does not contain the correct number of values: {}".format(param_number))
    else:
        upper_bound = in_upper

    # lower bounds
    if in_lower is None:
        lower_bound = [0.01] * param_number
    elif len(in_lower) != param_number:
        raise ValueError(
            "Lower bound set for parameters does not contain the correct number of values: {}".format(param_number))
    else:
        lower_bound = in_lower

    return params, upper_bound, lower_bound


def parse_opt_settings(rounds, reps=None, maxiters=None, folds=None):
    """    
    Function to correctly deal with replicate numbers, maxiter and fold args.
    
    Arguments
    rounds: number of optimization rounds to perform
    reps: a list of integers controlling the number of replicates in each of three optimization rounds
    maxiters: a list of integers controlling the maxiter argument in each of three optimization rounds
    folds: a list of integers controlling the fold argument when perturbing input parameter values
    """
    rounds = int(rounds)

    # rep set
    # create scheme where final replicates will be 20, and all previous 10
    if reps is None:
        if rounds >= 2:
            reps_list = [10] * (rounds - 1)
            reps_list.insert(len(reps_list), 20)
        else:
            reps_list = [10] * rounds
    elif len(reps) != rounds:
        raise ValueError("List length of replicate values does match the number of rounds: {}".format(rounds))
    else:
        reps_list = reps

    # maxiters
    if maxiters is None:
        maxiters_list = [5] * rounds
    elif len(maxiters) != rounds:
        raise ValueError("List length of maxiter values does match the number of rounds: {}".format(rounds))
    else:
        maxiters_list = maxiters

    # folds
    # create scheme so if rounds is greater than three, will always end with two fold and then one fold
    if folds is None:
        if rounds >= 3:
            folds_list = [3] * (rounds - 2)
            folds_list.insert(len(folds_list), 2)
            folds_list.insert(len(folds_list), 1)
        elif rounds == 2:
            folds_list = [2] * (rounds - 1)
            folds_list.insert(len(folds_list), 1)
        else:
            folds_list = [2] * rounds
    elif len(folds) != rounds:
        raise ValueError("List length of fold values does match the number of rounds: {}".format(rounds))
    else:
        folds_list = folds

    return reps_list, maxiters_list, folds_list


def collect_results(fs, sim_model, params_opt, rep_info, fs_folded):
    """    
    Gather up a bunch of results, return a list with following elements: 
    [round_num_rep_num, log-likelihood, AIC, chi^2 test stat, theta, parameter values]
    
    Arguments
    fs: spectrum object name
    sim_model: model fit with optimized parameters
    params_opt: list of the optimized parameters
    fs_folded: a Boolean (True, False) for whether empirical spectrum is folded or not
    """

    # calculate theta
    theta = dadi.Inference.optimal_sfs_scaling(sim_model, fs)
    theta = np.around(theta, 2)

    # calculate likelihood
    ll = dadi.Inference.ll_multinom(sim_model, fs)
    ll = np.around(ll, 2)

    # calculate AIC
    aic = (-2 * (float(ll))) + (2 * len(params_opt))

    # get Chi^2
    scaled_sim_model = sim_model * theta
    if fs_folded is True:
        # calculate Chi^2 statistic for folded
        folded_sim_model = scaled_sim_model.fold()
        chi2 = np.sum((folded_sim_model - fs) ** 2 / folded_sim_model)
        chi2 = np.around(chi2, 2)
    elif fs_folded is False:
        # calculate Chi^2 statistic for unfolded
        chi2 = np.sum((scaled_sim_model - fs) ** 2 / scaled_sim_model)
        chi2 = np.around(chi2, 2)

    # store key results in temporary sublist, append to larger results list
    out = pd.Series({
        'round': rep_info['round'],
        'replicate': rep_info['rep'],
        'log_likelihood': ll,
        'aic': aic,
        'chi_sq': chi2,
        'theta': theta,
        'optimized_params': ','.join([str(x) for x in params_opt]),
        'replicate_time': int((dt.now() - rep_info['start_time']).total_seconds())
    })

    ret = pd.DataFrame(columns=['round', 'replicate', 'log_likelihood', 'aic', 'chi_sq', 'theta', 'optimized_params', 'replicate_time'])
    ret = ret.append(out, ignore_index=True)

    return ret


def write_log(outfile, model_name, rep_results, roundrep):
    """    
    Reproduce replicate log to bigger log file, because constantly re-written.
    
    Arguments
    outfile: prefix for output naming
    model_name: a label to slap on the output files; ex. "no_mig"
    rep_results: the list returned by collect_results function: 
                 [roundnum_repnum, log-likelihood, AIC, chi^2 test stat, theta, parameter values]
    roundrep: name of replicate (ex, "Round_1_Replicate_10")
    """
    temp_logname = "{}_log.txt".format(model_name)
    final_logname = "{0}_{1}_log.txt".format(outfile, model_name)

    with open(final_logname, 'a') as fh_log:
        fh_log.write("\n{}\n".format(roundrep))
        try:
            with open(temp_logname, 'r') as fh_templog:
                for line in fh_templog:
                    fh_log.write(line)
        except IOError:
            print("Nothing written to log file this replicate...")

        fh_log.write("likelihood = {}\n".format(rep_results[1]))
        fh_log.write("theta = {}\n".format(rep_results[4]))
        fh_log.write("Optimized parameters = {}\n".format(rep_results[5]))


def Optimize_Routine(fs_filename, pts, outfile, model_name, max_processes, rounds, param_number, fs_folded=True,
                     reps=None, maxiters=None, folds=None, in_params=None,
                     in_upper=None, in_lower=None, param_labels=None, optimizer="log_fmin"):
    """
    Main function for running dadi routine.

    Mandatory/Positional Arguments
    (1) fs_filename:  spectrum object filepath
    (2) pts: grid size for extrapolation, list of three values
    (3) outfile:  prefix for output naming
    (4) model_name: a label to slap on the output files; ex. "no_mig"
    (5) max_processes: number of processes available for multiprocessing
    (6) rounds: number of optimization rounds to perform
    (7) param_number: number of parameters in the model selected (can count in params line for the model)
    (8) fs_folded: A Boolean value (True or False) indicating whether the empirical fs is folded (True) or not (False). Default is True.

    Optional Arguments
    (9) reps: a list of integers controlling the number of replicates in each of three optimization rounds
    (10) maxiters: a list of integers controlling the maxiter argument in each of three optimization rounds
    (11) folds: a list of integers controlling the fold argument when perturbing input parameter values
    (12) in_params: a list of parameter values 
    (13) in_upper: a list of upper bound values
    (14) in_lower: a list of lower bound values
    (15) param_labels: a string, labels for parameters that will be written to the output file to keep track of their order
    (16) optimizer: a string, to select the optimizer. Choices include: log (BFGS method), 
                    log_lbfgsb (L-BFGS-B method), log_fmin (Nelder-Mead method, DEFAULT), and log_powell (Powell's method).
    """

    # call function that determines if our params and bounds have been set or need to be generated for us
    params, upper_bound, lower_bound = parse_params(param_number, in_params, in_upper, in_lower)

    # call function that determines if our replicates, maxiter, and fold have been set or need to be generated for us
    reps_list, maxiters_list, folds_list = parse_opt_settings(rounds, reps, maxiters, folds)

    # start keeping track of time it takes to complete optimizations for this model
    tbr = dt.now()

    # optimizer dict
    optdict = {"log": "BFGS method", "log_lbfgsb": "L-BFGS-B method", "log_fmin": "Nelder-Mead method",
               "log_powell": "Powell's method"}

    # We need an output file that will store all summary info for each replicate, across rounds
    output_file = "{0}_{1}_optimized.csv".format(outfile, model_name)
    detail_file = "{0}_{1}_detail.csv".format(outfile, model_name)

    optimized = pd.DataFrame(columns=[
        'round', 'replicate', 'log_likelihood', 'aic', 'chi_sq', 'theta', 'optimized_params', 'replicate_time'
    ])

    detail = pd.DataFrame(
        columns=['round', 'rep', 'iter', 'log_likelihood', 'input_params', 'opt_params']
    )

    py_exe = sys.executable
    script_path = os.path.join(Path(__file__).resolve().parent, 'Replicate.py')

    # for every round, execute the assigned number of replicates with other round-defined args (maxiter, fold, best_params)
    rounds = int(rounds)
    for r in range(rounds):
        # make sure first round params are assigned (either user input or auto generated)
        if r == 0:
            best_params = params
        # and that all subsequent rounds use the params from a previous best scoring replicate
        else:
            best_params = opt_params

        tasks = []
        running_tasks = []

        count_running = 0
        count_todo = reps_list[r]
        count_complete = 0

        # perform an optimization routine for each rep number in this round number
        for rep in range(1, (reps_list[r] + 1)):
            rt = Replicate_Task(
                py_exe, script_path, r + 1, rep, model_name, fs_filename, best_params, pts[r], folds[r], fs_folded,
                ','.join([str(x) for x in upper_bound]), ','.join([str(x) for x in lower_bound]), optimizer, maxiters_list[r],
                outfile + '_{}_{}'.format(r+1, rep)
            )

            tasks.append(rt)

        while True:
            count_just_completed = 0
            files_to_drop = []

            for task in running_tasks:
                if task.status() == Replicate_Task.SUCCESS:
                    count_just_completed += 1

                    detail = detail.append(pd.read_csv(task.detail_path))
                    optimized = optimized.append(pd.read_csv(task.optimized_path))

                    files_to_drop.append(task.detail_path)
                    files_to_drop.append(task.optimized_path)
                    files_to_drop.append(task.log_path)

                    print('Task Success: {}'.format(str(task)))
                elif task.status() == Replicate_Task.FAILED:
                    count_just_completed += 1
                    print('Task Failure: {}'.format(str(task)))
                elif task.status() == Replicate_Task.RUNNING:
                    print('Task Running: {}'.format(str(task)))

            running_tasks = [s for s in running_tasks if not s.is_complete()]

            count_complete += count_just_completed
            count_running -= count_just_completed

            for f in files_to_drop:
                if os.path.exists(f):
                    os.remove(f)

            # while there are available processes & reps still to do
            while (count_running < max_processes) & (count_todo > 0):
                tasks[count_running + count_complete].start()
                running_tasks.append(tasks[count_running + count_complete])

                count_running += 1
                count_todo -= 1

            if count_running + count_todo == 0:
                break

            time.sleep(5)

        # Now that this round is over, find the params with the lowest log likelihood
        # we'll use the parameters from the best rep to start the next round as the loop continues
        opt_params = optimized.loc[optimized['log_likelihood'] == optimized['log_likelihood'].min(), 'optimized_params'].iloc[0]
        opt_params = opt_params.split(',')

    # Now that all rounds are over, calculate elapsed time for the whole model
    print(
        "\nAnalysis Time for Model '{0}': {1} (H:M:S)\n\n".format(model_name, dt.now() - tbr) +
        "============================================================================"
    )

    optimized.to_csv(output_file, index=False)
    detail.to_csv(detail_file, index=False)

    return


def main():
    parser = ArgumentParser(description='Runs Multiprocessing Optimize Routine of dadi_pipeline')

    parser.add_argument('--input', help='Input File Path', type=str)
    parser.add_argument('--output', help='Output Prefix', type=str)

    parser.add_argument('--processes', help='Max Processes', type=int)
    parser.add_argument('--model', help='Model Name', type=str)
    parser.add_argument('--count_params', help='Count Model Parameters', type=int)

    parser.add_argument('--pts', help='Points', type=str)
    parser.add_argument('--reps', help='Reps', type=str)
    parser.add_argument('--max_iters', help='Max Iterations', type=str)
    parser.add_argument('--folds', help='Folds', type=str)

    args = parser.parse_args()
    input_path = args.input
    output_prefix = args.output

    max_processes = args.processes
    model_name = args.model
    count_params = args.count_params

    pts = [int(x) for x in args.pts.split(',')]
    reps = [int(x) for x in args.reps.split(',')]
    max_iters = [int(x) for x in args.max_iters.split(',')]
    folds = [int(x) for x in args.folds.split(',')]

    if (len(pts) != len(reps)) or (len(pts) != len(max_iters)) or (len(pts) != len(folds)):
        raise Exception('Invalid Inputs. Lengths Do Not Match')

    Optimize_Routine(
        input_path,
        pts, output_prefix, model_name, max_processes, len(reps), count_params,
        fs_folded=False, reps=reps, maxiters=max_iters, folds=folds
    )

    return 0

if __name__ == '__main__':
    main()
