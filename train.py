import argparse
import os
import sys
import time

from tqdm import tqdm
import torch
import torchnet.meter as meter
import torch.nn.parallel
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from modules.core.dataloader import PileupDataset, TextColor
from modules.models.ModelHandler import ModelHandler
"""
Train a model and save the model that performs best.

Input:
- A train CSV containing training image set information (usually chr1-18)
- A test CSV containing testing image set information (usually chr19)

Output:
- A trained model
"""


def test(test_file, batch_size, gpu_mode, trained_model, num_classes, num_workers):
    """
    Test a trained model
    :param test_file: Test CSV file containing the test set
    :param batch_size: Batch size for prediction
    :param gpu_mode: If True GPU will be used
    :param trained_model: Trained model
    :param num_classes: Number of output classes (3- HOM, HET, HOM_ALT)
    :param num_workers: Number of workers for data loader
    :return:
    """
    transformations = transforms.Compose([transforms.ToTensor()])

    # data loader
    validation_data = PileupDataset(test_file, transformations)
    validation_loader = DataLoader(validation_data,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=num_workers
                                   )

    # set the evaluation mode of the model
    test_model = trained_model.eval()
    if gpu_mode:
        test_model = test_model.cuda()

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Test the Model
    total_loss = 0
    total_images = 0
    accuracy = 0
    confusion_matrix = meter.ConfusionMeter(num_classes)
    with torch.no_grad():
        with tqdm(total=len(validation_loader), desc='Accuracy: ', leave=True, ncols=50) as pbar:
            for i, (images, labels, records) in enumerate(validation_loader):
                if gpu_mode:
                    images = images.cuda()
                    labels = labels.cuda()

                # Predict + confusion_matrix + loss
                outputs = test_model(images)
                confusion_matrix.add(outputs.data, labels.data)

                # Progress bar update
                pbar.update(1)
                cm_value = confusion_matrix.value()
                denom = (cm_value.sum() - cm_value[0][0]) if (cm_value.sum() - cm_value[0][0]) > 0 else 1.0
                accuracy = 100.0 * (cm_value[1][1] + cm_value[2][2]) / denom
                pbar.set_description("Accuracy: " + str(accuracy))

                loss = criterion(outputs.contiguous().view(-1, num_classes), labels.contiguous().view(-1))

                # loss count
                total_loss += loss.item()
                total_images += (images.size(0))

    avg_loss = total_loss / total_images if total_images else 0
    sys.stderr.write(TextColor.BLUE + 'Test Loss: ' + str(avg_loss) + "\n" + TextColor.END)
    sys.stderr.write(TextColor.GREEN + "Confusion Matrix: \n" + str(confusion_matrix.conf) + "\n" + TextColor.END)

    return str(confusion_matrix.conf), avg_loss, accuracy


def train(train_file, test_file, batch_size, epoch_limit, gpu_mode, num_workers, retrain_model,
          retrain_model_path, model_output_dir, stats_output_dir, num_classes=3):
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
    :param num_classes: Number of output classes (3- HOM, HET, HOM_ALT)
    :return:
    """
    train_loss_logger = open(stats_output_dir + "train_loss.log", 'w')
    test_loss_logger = open(stats_output_dir + "test_loss.log", 'w')
    confusion_matrix_logger = open(stats_output_dir + "confusion_matrix.log", 'w')
    transformations = transforms.Compose([transforms.ToTensor()])

    sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)
    train_data_set = PileupDataset(train_file, transformations)
    train_loader = DataLoader(train_data_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=gpu_mode
                              )
    sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)\

    # 'weight_decay': 7.50100195037884e-06, 'learning_rate': 0.00023342801798666553
    model = ModelHandler.get_new_model(gpu_mode)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00023342801798666553, weight_decay=7.50100195037884e-06)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

    # print out total number of trainable parameters in model
    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("pytorch_total_params", pytorch_total_params)

    if retrain_model is True:
        if os.path.isfile(retrain_model_path) is False:
            sys.stderr.write(TextColor.RED + "ERROR: INVALID PATH TO RETRAIN PATH MODEL --retrain_model_path\n")
            exit(1)
        sys.stderr.write(TextColor.GREEN + "INFO: RETRAIN MODEL LOADING\n" + TextColor.END)
        model = ModelHandler.load_model_for_training(model, retrain_model_path)
        optimizer = ModelHandler.load_optimizer(optimizer, retrain_model_path, gpu_mode)
        sys.stderr.write(TextColor.GREEN + "INFO: RETRAIN MODEL LOADED SUCCESSFULLY\n" + TextColor.END)

    if gpu_mode:
        model = model.cuda()

    # Loss function
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0

    if gpu_mode:
        model = torch.nn.DataParallel(model).cuda()
        criterion = criterion.cuda()

    # Train the Model
    sys.stderr.write(TextColor.PURPLE + 'Training starting\n' + TextColor.END)
    for epoch in range(start_epoch, epoch_limit, 1):
        total_loss = 0
        total_images = 0
        batch_no = 0

        model = model.train()

        sys.stderr.write(TextColor.PURPLE + 'Epoch: ' + str(epoch + 1) + "\n" + TextColor.END)
        with tqdm(total=len(train_loader), desc='Loss', leave=True, ncols=50) as progress_bar:
            for i, (images, labels, records) in enumerate(train_loader):
                if gpu_mode:
                    images = images.cuda()
                    labels = labels.cuda()

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs.contiguous().view(-1, num_classes), labels.contiguous().view(-1))
                loss.backward()
                optimizer.step()

                # loss count
                total_loss += loss.item()
                total_images += (images.size(0))

                # update progress bar
                avg_loss = (total_loss / total_images) if total_images else 0
                progress_bar.set_description("Loss: " + str(avg_loss))
                progress_bar.refresh()
                progress_bar.update(1)
                # save in log file
                train_loss_logger.write(str(epoch + 1) + "," + str(batch_no) + "," + str(avg_loss) + "\n")
                batch_no += 1

        progress_bar.close()
        save_best_model(model, optimizer, model_output_dir + "_epoch_" + str(epoch + 1))

        confusion_matrix, test_loss, accuracy = \
            test(test_file, batch_size, gpu_mode, model, num_classes, num_workers)

        # update the loggers
        test_loss_logger.write(str(epoch + 1) + "," + str(test_loss) + "," + str(accuracy) + "\n")
        confusion_matrix_logger.write(str(epoch + 1) + "\n" + str(confusion_matrix) + "\n")
        train_loss_logger.flush()
        test_loss_logger.flush()
        confusion_matrix_logger.flush()

        lr_scheduler.step()

    sys.stderr.write(TextColor.PURPLE + 'Finished training\n' + TextColor.END)


def save_best_model(best_model, optimizer, file_name):
    """
    Save the best model
    :param best_model: A trained model
    :param optimizer: Optimizer
    :param file_name: Output file name
    :return:
    """
    sys.stderr.write(TextColor.BLUE + "SAVING MODEL.\n" + TextColor.END)
    # if os.path.isfile(file_name + '_model.pkl'):
    #     os.remove(file_name + '_model.pkl')
    if os.path.isfile(file_name + '_checkpoint.pkl'):
        os.remove(file_name + '_checkpoint.pkl')
    # torch.save(best_model, file_name + '_model.pkl')
    ModelHandler.save_checkpoint({
        'state_dict': best_model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, file_name + '_checkpoint.pkl')
    sys.stderr.write(TextColor.RED + "MODEL SAVED SUCCESSFULLY.\n" + TextColor.END)


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

    train(FLAGS.train_file, FLAGS.test_file, FLAGS.batch_size, FLAGS.epoch_size, FLAGS.gpu_mode,
          FLAGS.num_workers, FLAGS.retrain_model, FLAGS.retrain_model_path, model_out_dir, stats_out_dir)
