from __future__ import print_function
import argparse
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
import pickle
import sys
import os
import time

# Custom generator for our dataset
from modules.hyperband.hyperband import Hyperband
from modules.handlers.TextColor import TextColor
from modules.hyperband.train import train
from modules.hyperband.test import test


class WrapHyperband:
    # Paramters of the model
    # depth=28 widen_factor=4 drop_rate=0.0
    def __init__(self, train_file, test_file, gpu_mode, model_out_dir, log_directory):
        self.space = {
            # Returns a value drawn according to exp(uniform(low, high)) so that the logarithm of the return value is
            # uniformly distributed.
            'learning_rate': hp.loguniform('lr', -12, -7),
            'l2': hp.loguniform('l2', -14, -7),
        }
        self.train_file = train_file
        self.test_file = test_file
        self.gpu_mode = gpu_mode
        self.log_directory = log_directory
        self.model_out_dir = model_out_dir

    def get_params(self):
        return sample(self.space)

    def try_params(self, n_iterations, params):
        # Number of iterations or epoch for the model to train on
        n_iterations = int(round(n_iterations))
        print(params)
        sys.stderr.write(TextColor.BLUE + ' Loss: ' + str(n_iterations) + "\n" + TextColor.END)
        sys.stderr.write(TextColor.BLUE + str(params) + "\n" + TextColor.END)

        batch_size = 512
        num_workers = 32
        epoch_limit = n_iterations
        lr = params['learning_rate']
        l2 = params['l2']

        model, optimizer = train(self.train_file, batch_size, epoch_limit, self.gpu_mode, num_workers, lr, l2)
        stats_dictionary = test(self.test_file, batch_size, self.gpu_mode, model, num_workers)
        return model, optimizer, stats_dictionary

    def run(self, save_output):
        hyperband = Hyperband(self.get_params, self.try_params, max_iteration=8, downsample_rate=2,
                              model_directory=self.model_out_dir, log_directory=self.log_directory)
        results = hyperband.run()

        if save_output:
            with open('results.pkl', 'wb') as f:
                pickle.dump(results, f)

        # Print top 5 configs based on loss
        results = sorted(results, key=lambda r: r['loss'])[:5]
        print(results)


def handle_output_directory(output_dir):
    """
    Process the output directory and return a valid directory where we save the output
    :param output_dir: Output directory path
    :return:
    """
    # process the output directory
    if output_dir[-1] != "/":
        output_dir += "/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # create an internal directory so we don't overwrite previous runs
    timestr = time.strftime("%m%d%Y_%H%M%S")
    internal_directory = "hyperband_run_" + timestr + "/"
    output_dir = output_dir + internal_directory

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    model_output_dir = output_dir+'trained_models/'
    log_dir = output_dir+'logs/'
    os.mkdir(model_output_dir)
    os.mkdir(log_dir)

    return model_output_dir, log_dir


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks to generate the pileup.
    '''
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Training data description csv file."
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Training data description csv file."
    )
    parser.add_argument(
        "--gpu_mode",
        type=bool,
        default=False,
        help="If true then cuda is on."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default='./hyperband_output/',
        help="Directory to save the model"
    )
    FLAGS, unparsed = parser.parse_known_args()
    model_output_dir, log_dir = handle_output_directory(FLAGS.output_dir)
    wh = WrapHyperband(FLAGS.train_file, FLAGS.test_file, FLAGS.gpu_mode, model_output_dir, log_dir)
    wh.run(save_output=True)
