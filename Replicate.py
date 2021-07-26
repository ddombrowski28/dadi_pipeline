from argparse import ArgumentParser
import dadi
import pandas as pd
import os
from datetime import datetime as dt

from Model_Mapping import MODEL_MAPPING
from Optimize_Functions import collect_results


def replicate(model_name, round_num, rep_num, input_params, input_path, pts, folds, upper_bound, lower_bound, optimizer, max_iters, fs_folded, output_prefix):
    data_file = output_prefix + '.txt'
    detail_output = output_prefix + '_detail.csv'
    optimized_output = output_prefix + '_optimized.csv'

    # pull in data from input path
    fs = dadi.Spectrum.from_file(input_path)

    if fs_folded.lower() == 'true':
        fs_folded = True
    elif fs_folded.lower() == 'false':
        fs_folded = False
    else:
        raise Exception('Invalid FS Folded Value: {}'.format(fs_folded))
    st_time = dt.now()

    l_bound = [float(x) for x in lower_bound.split(',')]
    u_bound = [float(x) for x in upper_bound.split(',')]


    # create an extrapolating function
    func_exec = dadi.Numerics.make_extrap_log_func(MODEL_MAPPING[model_name])

    # perturb starting parameters
    params_perturbed = dadi.Misc.perturb_params(
        input_params, fold=folds,
        upper_bound=u_bound, lower_bound=l_bound
    )

    # optimize from perturbed parameters
    opt_funcs = {
        'log_fmin': dadi.Inference.optimize_log_fmin,
        'log': dadi.Inference.optimize_log,
        'log_lbfgsb': dadi.Inference.optimize_log_lbfgsb,
        'log_powell': dadi.Inference.optimize_log_powell
    }
    opt_func = opt_funcs.get(optimizer)
    if not opt_func:
        raise Exception(
            "ERROR: Unrecognized optimizer option: {}".format(optimizer) +
            "Please select from: log, log_lbfgsb, log_fmin, or log_powell."
        )

    params_opt = opt_func(
        params_perturbed, fs, func_exec, pts,
        lower_bound=l_bound, upper_bound=u_bound,
        verbose=1, maxiter=max_iters, output_file=data_file
    )

    detail = pd.DataFrame(
        columns=['round', 'rep', 'iter', 'log_likelihood', 'input_params', 'opt_params']
    )

    with open(data_file, 'r') as log_read:
        for line in log_read:
            if line == '\n':
                continue

            line_elements = [x.replace('array([', '').replace('])', '').strip() for x in line.split(',', 2)]

            append_series = pd.Series({
                'round': round_num,
                'rep': rep_num,
                'iter': int(line_elements[0]),
                'log_likelihood': float(line_elements[1]),
                'input_params': input_params,
                'opt_params': [float(x.strip()) for x in line_elements[2].split(',')]
            })

            detail = detail.append(append_series, ignore_index=True)
    os.remove(data_file)

    # simulate the model with the optimized parameters
    sim_model = func_exec(params_opt, fs.sample_sizes, pts)

    # collect results and store to pandas
    rep_results = collect_results(fs, sim_model, params_opt, {'round': round_num, 'rep': rep_num, 'start_time': st_time, 'input_params': input_params}, fs_folded)

    detail.to_csv(detail_output, index=False)
    rep_results.to_csv(optimized_output, index=False)

    return

def main():
    parser = ArgumentParser(description='Runs Individual Replicate for Multiprocessing')

    parser.add_argument('--model', help='Model', type=str)
    parser.add_argument('--round', help='Round', type=int)
    parser.add_argument('--rep', help='Rep', type=int)
    parser.add_argument('--params', help='Params', type=str)
    parser.add_argument('--input_path', help='Input Path', type=str)
    parser.add_argument('--pts', help='Points', type=int)
    parser.add_argument('--folds', help='Folds', type=int)
    parser.add_argument('--fs_folded', help= 'FS Folded?', type = str)
    parser.add_argument('--upper', help='Upper Bound', type=str)
    parser.add_argument('--lower', help='Lower Bound', type=str)
    parser.add_argument('--optimizer', help='Optimizer', type=str)
    parser.add_argument('--max_iters', help='Max Iterations', type=int)
    parser.add_argument('--output_prefix', help='Output File Prefix', type=str)

    args = parser.parse_args()

    replicate(
        args.model, args.round, args.rep, [float(x) for x in args.params.split(',')], args.input_path, args.pts, args.folds,
        args.upper, args.lower, args.optimizer, args.max_iters, args.fs_folded, args.output_prefix
    )

    return 0


if __name__ == '__main__':
    main()
