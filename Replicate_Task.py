"""
Author: Derek Dombrowski
Date: 2021-07-24
Purpose: Class to be used by multiprocessing routine
    Connect & informs top process of subprocess status
"""

import os
import subprocess
import threading
from datetime import datetime as dt
import time
import pandas as pd


class Replicate_Task:
    # Task States
    FAILED = 'failed'
    SUCCESS = 'success'
    RUNNING = 'running'
    NOT_LAUNCHED = 'not launched'

    # Formats
    LOG_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

    def __init__(self, python_exe, path, round_num, rep_num, model, input_file, input_params, pts, folds, fs_folded, upper, lower, optimizer, max_iters, output_path):
        self.python_exe = python_exe
        self.path = path

        self.input = input_file
        self.model = model
        self.round = round_num
        self.rep = rep_num
        self.input_params = input_params

        self.pts = pts
        self.max_iterations = max_iters
        self.folds = folds
        self.fs_folded = fs_folded
        self.upper_bound = upper
        self.lower_bound = lower
        self.output = output_path
        self.optimizer = optimizer

        self.task_result = self.NOT_LAUNCHED

        self.log_path = self.output + '.log'
        self.detail_path = self.output + '_detail.csv'
        self.optimized_path = self.output + '_optimized.csv'

        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = dt.now()

        while not self.start_time:
            time.sleep(0.1)

        thread = threading.Thread(target=self.run, name='Replicate_Task', args=(), daemon=False)
        thread.start()

        while not self.task_result:
            time.sleep(0.1)

        return

    def run_duration(self):
        if not self.start_time:
            return 0
        else:
            if not self.end_time:
                return dt.now() - self.start_time
            else:
                return self.end_time - self.start_time

    def status(self):
        return self.task_result

    def __str__(self):
        ret = '%(model)s: [%(round)s:%(rep)s - %(pts)s/%(max_iterations)s/%(folds)s] ' % {
            'model': self.model, 'round': self.round, 'rep': self.rep,
            'pts': self.pts, 'max_iterations': self.max_iterations, 'folds': self.folds
        }
        ret += 'STATUS [%(status)s] RUN DURATION [%(dur)s]' % {
            'status': self.status(), 'dur': int(self.run_duration().total_seconds())
        }
        return ret

    def is_complete(self):
        if self.task_result != self.NOT_LAUNCHED:
            return self.task_result in [self.FAILED, self.SUCCESS]
        else:
            return False

    def create_arguments(self):
        arg_list = ['--model={}'.format(self.model), '--params={}'.format(','.join([str(x) for x in self.input_params])),
                    '--round={}'.format(self.round), '--rep={}'.format(self.rep),
                    '--input_path={}'.format(self.input), '--pts={}'.format(self.pts),
                    '--folds={}'.format(self.folds), '--fs_folded={}'.format(self.fs_folded),
                    '--upper={}'.format(self.upper_bound), '--lower={}'.format(self.lower_bound),
                    '--optimizer={}'.format(self.optimizer),
                    '--max_iters={}'.format(self.max_iterations), '--output_prefix={}'.format(self.output)]

        return arg_list

    def run(self):
        self.task_result = self.RUNNING

        arg_list = self.create_arguments()
        arg_string = ' '.join(arg_list)

        print('Subprocess launching [%(exe)s %(path)s %(arg_string)s] at [%(time)s]' % {
           'arg_string': arg_string, 'time': self.start_time.strftime(self.LOG_TIME_FORMAT),
           'exe': self.python_exe, 'path': self.path
        })

        with open(self.output + '.log', 'a') as data_out:
            process_result = subprocess.run([self.python_exe, self.path, *arg_list], stderr=data_out, stdout=data_out)

        print('Subprocess Return Code [{}]'.format(process_result.returncode))
        print('Subprocess Complete')

        if process_result.returncode == 0:
            self.task_result = self.SUCCESS
        else:
            self.task_result = self.FAILED
        return
