from __future__ import print_function
import numpy as np
from random import random
from math import log, ceil
from time import time, ctime
import sys
from modules.handlers.TextColor import TextColor
import logging
import os
from datetime import datetime
import torch
from modules.models.ModelHandler import ModelHandler
"""
The Hyperband class used for hyper-parameter optimization.
Optimized from:
https://github.com/zygmuntz/hyperband
"""


def save_best_model_hyperband(best_model, optimizer, file_name):
    """
    Save the best model
    :param best_model: A trained model
    :param optimizer: Optimizer
    :param file_name: Output file name
    :return:
    """
    sys.stderr.write(TextColor.BLUE + "SAVING MODEL.\n" + TextColor.END)
    if os.path.isfile(file_name + '_model.pkl'):
        os.remove(file_name + '_model.pkl')
    if os.path.isfile(file_name + '_checkpoint.pkl'):
        os.remove(file_name + '_checkpoint.pkl')
    torch.save(best_model, file_name + '_model.pkl')
    ModelHandler.save_checkpoint({
        'state_dict': best_model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, file_name + '_checkpoint.pkl')
    sys.stderr.write(TextColor.RED + "MODEL SAVED SUCCESSFULLY.\n" + TextColor.END)


class Hyperband:
    """
    Hyper-parameter optimization algorithm implemented
    This class is optimized from the popular hyperband implementation:
    https://github.com/zygmuntz/hyperband
    """
    def __init__(self, get_params_function, try_params_function, max_iteration, downsample_rate, log_directory,
                 model_directory):
        """
        Initialize hyperband object.
        :param get_params_function: A function to get parameters from sample space.
        :param try_params_function: Try a parameter by training and testing it
        :param max_iteration: Maximum iterations per configuration
        :param downsample_rate: Defines configuration downsampling rate
        :param log_directory: Directory where log is saved
        :param model_directory: Directory where model is saved
        """
        self.get_params = get_params_function
        self.try_params = try_params_function

        self.max_iter = max_iteration  # maximum iterations per configuration
        self.eta = downsample_rate  # defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter

        self.results = []  # list of dicts
        self.counter = 0
        self.best_loss = np.inf
        self.best_acc = 0
        self.best_counter = -1
        self.log_file = log_directory+'Hyperband_'+datetime.now().strftime("%Y%m%d-%H%M%S")+'.log'
        self.model_directory = model_directory + 'Hyperband_trained_' + datetime.now().strftime("%Y%m%d-%H%M%S")

        logging.basicConfig(filename=self.log_file, level=logging.INFO)

    # can be called multiple times
    def run(self, skip_last=0, dry_run=False):
        """
        The hyper-parameter optimization algorithm.
        :param skip_last: Skip the last iteration
        :param dry_run: Dry test run if True
        :return:
        """
        for s in reversed(range(self.s_max + 1)):

            # initial number of configurations
            n = int(ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))

            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s)

            # n random configurations
            T = [self.get_params() for i in range(n)]

            for i in range((s + 1) - int(skip_last)):  # changed from s + 1

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations

                n_configs = n * self.eta ** (-i)
                n_iterations = r * self.eta ** (i)

                sys.stderr.write(TextColor.BLUE + "\n*** {} configurations x {:.5f} iterations each"
                                 .format(n_configs, n_iterations) + "\n" + TextColor.END)

                logging.info("\n*** {} configurations x {:.1f} iterations each".format(n_configs, n_iterations))

                val_losses = []
                early_stops = []

                for t in T:

                    self.counter += 1
                    sys.stderr.write(TextColor.BLUE + "{} | {} | lowest loss so far: {} | (run {})"
                                     .format(self.counter, ctime(), self.best_loss, self.best_counter) + TextColor.END)
                    logging.info("{} | {} | lowest loss so far: {} | (run {})"
                                 .format(self.counter, ctime(), self.best_loss, self.best_counter))

                    start_time = time()

                    if dry_run:
                        result = {'loss': random(), 'log_loss': random(), 'auc': random()}
                        model = ''
                        optimizer = ''
                    else:
                        logging.info("Iterations:\t" + str(n_iterations))
                        logging.info("Params:\t" + str(t))
                        model, optimizer, result = self.try_params(n_iterations, t)  # <---

                    assert (type(result) == dict)
                    assert ('loss' in result)

                    seconds = int(round(time() - start_time))
                    sys.stderr.write(TextColor.BLUE + "\n{} seconds.\n".format(seconds) + TextColor.END)

                    loss = result['loss']
                    val_losses.append(loss)

                    early_stop = result.get('early_stop', False)
                    early_stops.append(early_stop)

                    # keeping track of the best result so far (for display only)
                    # could do it be checking results each time, but hey
                    if loss < self.best_loss:
                        save_best_model_hyperband(model, optimizer, self.model_directory)
                        self.best_loss = loss
                        self.best_counter = self.counter

                    result['counter'] = self.counter
                    result['seconds'] = seconds
                    result['params'] = t
                    result['iterations'] = n_iterations

                    self.results.append(result)

                # select a number of best configurations for the next loop
                # filter out early stops, if any
                indices = np.argsort(val_losses)
                T = [T[i] for i in indices if not early_stops[i]]
                T = T[0:int(n_configs / self.eta)]

        return self.results
