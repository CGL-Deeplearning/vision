import argparse
import os
import sys
import time

from modules.core.dataloader import TextColor
from modules.models.ModelHandler import ModelHandler
from modules.models.train import train
"""
Train a model and save the model that performs best.

Input:
- A train CSV containing training image set information (usually chr1-18)
- A test CSV containing testing image set information (usually chr19)

Output:
- A trained model
"""


def call_train(train_file, test_file, batch_size, epoch_limit, gpu_mode, num_workers, retrain_model,
               retrain_model_path, model_output_dir, stats_output_dir, learning_rate, weight_decay):
    """
    Train a model and save
    :param train_file: A CSV file containing train image information
    :param test_file: A CSV file containing test image information
    :param batch_size: Batch size for training
    :param epoch_limit: Number of epochs to train on
    :param gpu_mode: If true the model will be trained on GPU
    :param num_workers: Number of workers for data loading
    :param retrain_model: If true then an existing trained model will be loaded and trained
    :param retrain_model_path: If retrain is true then this should be the path to an already trained model
    :param model_output_dir: Directory where model will be saved
    :param stats_output_dir: Directory where statistics of the model will be saved
    :param learning_rate: Learning rate of the optimizer
    :param weight_decay: Weight decay of the optimizer
    :return:
    """
    momentum = 0.9
    sys.stderr.write(TextColor.BLUE + "Learning rate: " + str(learning_rate) + "\n" + TextColor.END)
    sys.stderr.write(TextColor.BLUE + "Weight Decay: " + str(weight_decay) + "\n" + TextColor.END)
    model, optimizer, stats = train(train_file, test_file, batch_size,
                                    epoch_limit, 0, gpu_mode,
                                    num_workers, retrain_model, retrain_model_path,
                                    learning_rate, weight_decay, momentum,
                                    stats_output_dir, model_output_dir,
                                    hyperband_mode=False, num_classes=3)
    sys.stderr.write(TextColor.PURPLE + 'Finished training\n' + TextColor.END)


def handle_output_directory(output_dir):
    """
    Process the output directory and return a valid directory where we save the output
    :param output_dir: Output directory path
    :return:
    """
    timestr = time.strftime("%m%d%Y_%H%M%S")
    # process the output directory
    if output_dir[-1] != "/":
        output_dir += "/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # create an internal directory so we don't overwrite previous runs
    model_save_dir = output_dir + "trained_models_" + timestr + "/"
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    stats_directory = model_save_dir + "stats_" + timestr + "/"

    if not os.path.exists(stats_directory):
        os.mkdir(stats_directory)

    return model_save_dir, stats_directory


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks.
    '''
    parser = argparse.ArgumentParser()
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
        "--batch_size",
        type=int,
        required=False,
        default=100,
        help="Batch size for training, default is 100."
    )
    parser.add_argument(
        "--epoch_size",
        type=int,
        required=False,
        default=10,
        help="Epoch size for training iteration."
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        default=0.001,
        help="Learning rate for training."
    )
    parser.add_argument(
        "--l2",
        type=float,
        required=False,
        default=0.0001,
        help="Weight decay for training."
    )
    parser.add_argument(
        "--model_out",
        type=str,
        required=False,
        default='./model',
        help="Path and file_name to save model, default is ./model"
    )
    parser.add_argument(
        "--retrain_model",
        type=bool,
        default=False,
        help="If true then retrain a pre-trained mode."
    )
    parser.add_argument(
        "--retrain_model_path",
        type=str,
        required=False,
        default='./model',
        help="Path of the model to load and retrain"
    )
    parser.add_argument(
        "--gpu_mode",
        type=bool,
        default=False,
        help="If true then cuda is on."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=40,
        help="Epoch size for training iteration."
    )

    FLAGS, unparsed = parser.parse_known_args()
    model_out_dir, stats_out_dir = handle_output_directory(FLAGS.model_out.rpartition('/')[0] + "/")
    model_out_dir = model_out_dir + "VISION_model"
    sys.stderr.write(TextColor.BLUE + "THE MODEL AND STATS LOCATION: " + str(model_out_dir) + "\n" + TextColor.END)

    call_train(FLAGS.train_file, FLAGS.test_file, FLAGS.batch_size, FLAGS.epoch_size, FLAGS.gpu_mode,
               FLAGS.num_workers, FLAGS.retrain_model, FLAGS.retrain_model_path, model_out_dir, stats_out_dir,
               FLAGS.lr, FLAGS.l2)
