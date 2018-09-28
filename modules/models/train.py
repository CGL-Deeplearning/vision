from __future__ import print_function
import os
import sys

from tqdm import tqdm
import torch
import torch.nn.parallel
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from modules.core.dataloader import PileupDataset, TextColor
from modules.core.data_sampler import BalancedSampler
from modules.models.ModelHandler import ModelHandler
from modules.models.test import test
"""
Train a model and return the model and optimizer trained.

Input:
- A train CSV containing training image set information (usually chr1-18)

Return:
- A trained model
"""


def train(train_file, test_file, batch_size, epoch_limit, prev_ite,
          gpu_mode, num_workers, retrain_model, retrain_model_path,
          learning_rate, weight_decay, momentum, stats_output_dir,
          model_output_dir, hyperband_mode, num_classes=3):

    if hyperband_mode is False:
        train_loss_logger = open(stats_output_dir + "train_loss.log", 'w')
        test_loss_logger = open(stats_output_dir + "test_loss.log", 'w')
        confusion_matrix_logger = open(stats_output_dir + "confusion_matrix.log", 'w')
    else:
        train_loss_logger = None
        test_loss_logger = None
        confusion_matrix_logger = None

    transformations = transforms.Compose([transforms.ToTensor()])
    print("Initializating dataset")
    train_data_set = PileupDataset(train_file, transformations)
    print("Dataset done")
    print("Initializing dataloader")
    train_loader = DataLoader(train_data_set,
                              sampler=BalancedSampler(train_data_set),
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=gpu_mode
                              )
    print("Initialization done")
    # this needs to change
    model = ModelHandler.get_new_model(gpu_mode)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

    if retrain_model is True:
        if os.path.isfile(retrain_model_path) is False:
            sys.stderr.write(TextColor.RED + "ERROR: INVALID PATH TO RETRAIN PATH MODEL --retrain_model_path\n")
            exit(1)
        # sys.stderr.write(TextColor.GREEN + "INFO: RETRAIN MODEL LOADING\n" + TextColor.END)
        model = ModelHandler.load_model_for_training(model, retrain_model_path)

        optimizer = ModelHandler.load_optimizer(optimizer, retrain_model_path, gpu_mode)
        sys.stderr.write(TextColor.GREEN + "INFO: RETRAIN MODEL LOADED\n" + TextColor.END)

    if gpu_mode:
        model = torch.nn.DataParallel(model).cuda()

    # Loss
    criterion = nn.CrossEntropyLoss()

    if gpu_mode is True:
        criterion = criterion.cuda()

    start_epoch = prev_ite

    stats = dict()
    stats['loss_epoch'] = []
    stats['accuracy_epoch'] = []

    for epoch in range(start_epoch, epoch_limit, 1):
        total_loss = 0
        total_images = 0
        sys.stderr.write(TextColor.BLUE + 'Train epoch: ' + str(epoch + 1) + "\n")
        # make sure the model is in train mode. BN is different in train and eval.
        model.train()

        batch_no = 1
        with tqdm(total=len(train_loader), desc='Loss', ncols=100) as progress_bar:
            for (images, labels, rec) in train_loader:
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
                batch_no += 1

                # update progress bar
                avg_loss = (total_loss / total_images) if total_images else 0
                progress_bar.set_description("Loss: " + str(avg_loss))
                # train_loss_logger.write(str(epoch + 1) + "," + str(batch_no) + "," + str(avg_loss) + "\n")
                if hyperband_mode is False:
                    train_loss_logger.write(str(epoch + 1) + "," + str(batch_no) + "," + str(avg_loss) + "\n")
                progress_bar.refresh()
                progress_bar.update(1)

            progress_bar.close()

        confusion_matrix, stats_dictionary = test(test_file, batch_size, gpu_mode, model, num_workers, num_classes=3)
        stats['loss'] = stats_dictionary['loss']
        stats['accuracy'] = stats_dictionary['accuracy']
        stats['loss_epoch'].append((epoch, stats_dictionary['loss']))
        stats['accuracy_epoch'].append((epoch, stats_dictionary['accuracy']))

        if hyperband_mode is False:
            test_loss_logger.write(str(epoch + 1) + "," + str(stats_dictionary['loss']) + "," +
                                   str(stats_dictionary['accuracy']) + "\n")
            confusion_matrix_logger.write(str(epoch + 1) + "\n" + str(confusion_matrix) + "\n")
            save_best_model(model, optimizer, model_output_dir)

        lr_scheduler.step()
        # if epoch > 3:
        #     for param_group in optimizer.param_groups:
        #         momentum_value = param_group['momentum']
        #         param_group['momentum'] = max(0.9, momentum_value + 0.1)

        if hyperband_mode is True and epoch + 1 >= 2 and stats_dictionary['accuracy'] <= 90.0:
            sys.stderr.write(TextColor.RED + 'EARLY STOPPING\n' + TextColor.END)
            break

    return model, optimizer, stats


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

