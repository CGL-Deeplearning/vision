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
from modules.models.ModelHandler import ModelHandler
from modules.hyperband.test import test
"""
Train a model and return the model and optimizer trained.

Input:
- A train CSV containing training image set information (usually chr1-18)

Return:
- A trained model
"""


def train(train_file, test_file, batch_size, epoch_limit, prev_ite, gpu_mode, num_workers, retrain_model,
          retrain_model_path, learning_rate, momentum, num_classes=3):
    transformations = transforms.Compose([transforms.ToTensor()])

    sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)
    train_data_set = PileupDataset(train_file, transformations)
    train_loader = DataLoader(train_data_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=gpu_mode
                              )
    # this needs to change
    model = ModelHandler.get_new_model(gpu_mode)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

    if retrain_model is True:
        if os.path.isfile(retrain_model_path) is False:
            sys.stderr.write(TextColor.RED + "ERROR: INVALID PATH TO RETRAIN PATH MODEL --retrain_model_path\n")
            exit(1)
        sys.stderr.write(TextColor.GREEN + "INFO: RETRAIN MODEL LOADING\n" + TextColor.END)
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

    # Train the Model
    sys.stderr.write(TextColor.PURPLE + 'Training starting\n' + TextColor.END)
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
        with tqdm(total=len(train_loader), desc='Loss', leave=True, ncols=100) as progress_bar:
            for (images, labels, rec) in train_loader:
                if gpu_mode:
                    images = images.cuda()
                    labels = labels.cuda()

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs.contiguous().view(-1, num_classes), y.contiguous().view(-1))
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
                progress_bar.refresh()
                progress_bar.update(1)
            progress_bar.close()

        stats_dictioanry = test(test_file, batch_size, gpu_mode, model, num_workers, num_classes=3)
        stats['loss'] = stats_dictioanry['loss']
        stats['accuracy'] = stats_dictioanry['accuracy']
        stats['loss_epoch'].append((epoch, stats_dictioanry['loss']))
        stats['accuracy_epoch'].append((epoch, stats_dictioanry['accuracy']))

        lr_scheduler.step()

        if epoch > 2 and stats_dictioanry['accuracy'] < 90:
            sys.stderr.write(TextColor.PURPLE + 'EARLY STOPPING\n' + TextColor.END)
            break

    sys.stderr.write(TextColor.PURPLE + 'Finished training\n' + TextColor.END)

    return model, optimizer, stats

