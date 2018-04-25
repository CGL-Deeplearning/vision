from __future__ import print_function
import sys
import torch
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import time

# Custom generator for our dataset
from torch.utils.data import DataLoader
from modules.core.dataloader import PileupDataset, TextColor
from modules.models.inception import Inception3

'''Train the model and return'''


def train(train_file, batch_size, epoch_limit, gpu_mode, num_workers, lr, wd, num_classes=3, debug_print=False):
    transformations = transforms.Compose([transforms.ToTensor()])

    # sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)
    train_data_set = PileupDataset(train_file, transformations)
    train_loader = DataLoader(train_data_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=gpu_mode
                              )
    # sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)

    model = Inception3()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if gpu_mode:
        model = model.cuda()

    # Loss function
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0

    if gpu_mode:
        model = torch.nn.DataParallel(model).cuda()

    # Train the Model
    # sys.stderr.write(TextColor.PURPLE + 'Training starting\n' + TextColor.END)
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

            if debug_print is True:
                avg_loss = (total_loss / total_images) if total_images else 0
                sys.stderr.write(TextColor.BLUE + "EPOCH: " + str(epoch+1) + " Batches done: " + str(batches_done)
                                 + " / " + str(len(train_loader)) + "\n" + TextColor.END)
                sys.stderr.write(TextColor.YELLOW + "Loss: " + str(avg_loss) + "\n" + TextColor.END)
                sys.stderr.write(TextColor.DARKCYAN + "Time Elapsed: " + str(time.time() - start_time) +
                                 "\n" + TextColor.END)
                start_time = time.time()

        avg_loss = (total_loss / total_images) if total_images else 0
        sys.stderr.write(TextColor.BLUE + "EPOCH " + str(epoch+1) + ": " + TextColor.END)
        sys.stderr.write(TextColor.YELLOW + "LOSS: " + str(avg_loss) + TextColor.END)
        sys.stderr.write(TextColor.DARKCYAN + " TIME: " + str(time.time() - epoch_start_time) +
                         "\n" + TextColor.END)

    # sys.stderr.write(TextColor.PURPLE + 'Finished training\n' + TextColor.END)

    return model, optimizer
