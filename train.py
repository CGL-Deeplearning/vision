import argparse
import os
import sys
import time

import torch
import torchnet.meter as meter
import torch.nn.parallel
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from modules.core.dataloader import PileupDataset, TextColor
from modules.models.ModelHandler import ModelHandler
from modules.models.inception import Inception3
"""
Train a model and save the model that performs best.

Input:
- A train CSV containing training image set information (usually chr1-18)
- A test CSV containing testing image set information (usually chr19)

Output:
- A trained model
"""


def test(data_file, batch_size, gpu_mode, trained_model, num_classes, num_workers):
    """
    Test a trained model
    :param data_file: Test CSV file containing the test set
    :param batch_size: Batch size for prediction
    :param gpu_mode: If True GPU will be used
    :param trained_model: Trained model
    :param num_classes: Number of output classes (3- HOM, HET, HOM_ALT)
    :param num_workers: Number of workers for data loader
    :return:
    """
    transformations = transforms.Compose([transforms.ToTensor()])

    # data loader
    validation_data = PileupDataset(data_file, transformations)
    validation_loader = DataLoader(validation_data,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=num_workers,
                                   pin_memory=gpu_mode
                                   )
    sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)

    # set the evaluation mode of the model
    test_model = trained_model.eval()
    if gpu_mode:
        test_model = test_model.cuda()

    # Loss
    test_criterion = nn.CrossEntropyLoss()

    # Test the Model
    sys.stderr.write(TextColor.PURPLE + 'Test starting\n' + TextColor.END)
    total_loss = 0
    total_images = 0
    batches_done = 0
    confusion_matrix = meter.ConfusionMeter(num_classes)
    with torch.no_grad():
        for i, (images, labels, records) in enumerate(validation_loader):
            if gpu_mode is True and images.size(0) % 8 != 0:
                continue

            images = Variable(images, volatile=True)
            labels = Variable(labels, volatile=True)
            if gpu_mode:
                images = images.cuda()
                labels = labels.cuda()

            # Predict + confusion_matrix + loss
            outputs = test_model(images)
            confusion_matrix.add(outputs.data, labels.data)
            test_loss = test_criterion(outputs.contiguous().view(-1, num_classes), labels.contiguous().view(-1))
            # Loss count
            total_loss += float(test_loss.data[0])
            total_images += float(images.size(0))
            batches_done += 1
            if batches_done % 10 == 0:
                sys.stderr.write(str(confusion_matrix.conf)+"\n")
                sys.stderr.write(TextColor.BLUE+'Batches done: ' + str(batches_done) + " / " + str(len(validation_loader)) +
                                 "\n" + TextColor.END)

    avg_loss = total_loss / total_images if total_images else 0
    print('Test Loss: ' + str(avg_loss))
    print('Confusion Matrix: \n', confusion_matrix.conf)
    # print summaries
    sys.stderr.write(TextColor.YELLOW+'Test Loss: ' + str(avg_loss) + "\n"+TextColor.END)
    sys.stderr.write("Confusion Matrix \n: " + str(confusion_matrix.conf) + "\n" + TextColor.END)


def train(train_file, validation_file, batch_size, epoch_limit, file_name, gpu_mode, num_workers, retrain_model,
          retrain_model_path, num_classes=3):
    """
    Train a model and save
    :param train_file: A CSV file containing train image information
    :param validation_file: A CSV file containing test image information
    :param batch_size: Batch size for training
    :param epoch_limit: Number of epochs to train on
    :param file_name: The model output file name
    :param gpu_mode: If true the model will be trained on GPU
    :param num_workers: Number of workers for data loading
    :param retrain_mode: If true then an existing trained model will be loaded and trained
    :param model_path: If retrain is true then this should be the path to an already trained model
    :param num_classes: Number of output classes (3- HOM, HET, HOM_ALT)
    :return:
    """
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

    model = Inception3()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00021723010296152584, weight_decay=1.4433597247180705e-06)

    #print out total number of trainable parameters in model
    #pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print("pytorch_total_params", pytorch_total_params)

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

    running_test_loss = -1

    # Train the Model
    sys.stderr.write(TextColor.PURPLE + 'Training starting\n' + TextColor.END)
    for epoch in range(start_epoch, epoch_limit, 1):
        total_loss = 0
        total_images = 0
        epoch_start_time = time.time()
        start_time = time.time()
        batches_done = 0
        for i, (images, labels, records) in enumerate(train_loader):
            if gpu_mode is True and images.size(0) % 8 != 0:
                continue

            images = Variable(images)
            labels = Variable(labels)
            if gpu_mode:
                images = images.cuda()
                labels = labels.cuda()

            x = images
            y = labels

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs.contiguous().view(-1, num_classes), y.contiguous().view(-1))
            loss.backward()
            optimizer.step()

            # loss count
            total_loss += loss.data[0]
            total_images += (x.size(0))
            batches_done += 1

            if batches_done % 10 == 0:
                avg_loss = (total_loss / total_images) if total_images else 0
                print(str(epoch + 1) + "\t" + str(i + 1) + "\t" + str(total_loss) + "\t" + str(avg_loss))
                #sys.stdout.flush()
                sys.stderr.write(TextColor.BLUE + "EPOCH: " + str(epoch+1) + " Batches done: " + str(batches_done)
                                 + " / " + str(len(train_loader)) + "\n" + TextColor.END)
                sys.stderr.flush()
                sys.stderr.write(TextColor.YELLOW + "Loss: " + str(avg_loss) + "\n" + TextColor.END)
                sys.stderr.write(TextColor.DARKCYAN + "Time Elapsed: " + str(time.time() - start_time) +
                                 "\n" + TextColor.END)
                start_time = time.time()
                sys.stderr.flush()

        avg_loss = (total_loss / total_images) if total_images else 0
        sys.stderr.write(TextColor.BLUE + "EPOCH: " + str(epoch+1) + " Completed: " + str(i+1) + "\n" + TextColor.END)
        sys.stderr.write(TextColor.YELLOW + "Loss: " + str(avg_loss) + "\n" + TextColor.END)
        sys.stderr.write(TextColor.DARKCYAN + "Time Elapsed: " + str(time.time() - epoch_start_time) +
                         "\n" + TextColor.END)
        sys.stderr.flush()
        print(str(epoch+1) + "\t" + str(i + 1) + "\t" + str(avg_loss))
        sys.stdout.flush()

        save_best_model(model, optimizer, file_name + "_epoch_" + str(epoch + 1))
        # After each epoch do validation
        test(validation_file, batch_size, gpu_mode, model, num_classes, num_workers)


        # optimizer = exp_lr_scheduler(optimizer, (epoch+1))

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


def directory_control(file_path):
    """
    Create a directory if it doesn't exist
    :param file_path: Path to the directory
    :return:
    """
    directory = os.path.dirname(file_path)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks.
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
    directory_control(FLAGS.model_out.rpartition('/')[0]+"/")
    train(FLAGS.train_file, FLAGS.test_file, FLAGS.batch_size, FLAGS.epoch_size, FLAGS.model_out, FLAGS.gpu_mode,
          FLAGS.num_workers, FLAGS.retrain_model, FLAGS.retrain_model_path)
