from __future__ import print_function
import sys
import torchnet.meter as meter
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from modules.core.dataloader import PileupDataset, TextColor
"""
This script will evaluate a model and return the loss value.

Input:
- A trained model
- A test CSV file to evaluate

Returns:
- Loss value
"""


def test(data_file, batch_size, gpu_mode, trained_model, num_workers, num_classes=3, debug_print=False):
    """
    Predict images of the data file and print names all the files that are missed in a CSV file.
    :param data_file: A CSV file containing image information
    :param batch_size: Size of the batch
    :param gpu_mode: If true the model will use GPUs
    :param trained_model: A trained model
    :param num_workers: Number of workers to use for loading data
    :param num_classes: Number of classes (HOM, HET, HOM_ALT)
    :param debug_print: If true the debug messages will print
    :return: Loss value
    """
    transformations = transforms.Compose([transforms.ToTensor()])

    validation_data = PileupDataset(data_file, transformations)
    validation_loader = DataLoader(validation_data,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=num_workers,
                                   pin_memory=gpu_mode
                                   )
    # sys.stderr.write(TextColor.PURPLE + 'Test data loading finished\n' + TextColor.END)

    test_model = trained_model.eval()
    if gpu_mode:
        test_model = test_model.cuda()

    # Loss
    test_criterion = nn.CrossEntropyLoss()

    # Test the Model
    # sys.stderr.write(TextColor.PURPLE + 'Test starting\n' + TextColor.END)
    total_loss = 0
    total_images = 0
    batches_done = 0
    confusion_matrix = meter.ConfusionMeter(num_classes)
    for i, (images, labels, records) in enumerate(validation_loader):
        if gpu_mode is True and images.size(0) % 8 != 0:
            continue

        images = Variable(images, volatile=True)
        labels = Variable(labels, volatile=True)
        if gpu_mode:
            images = images.cuda()
            labels = labels.cuda()

        # Forward + Backward + Optimize
        outputs = test_model(images)
        confusion_matrix.add(outputs.data, labels.data)
        test_loss = test_criterion(outputs.contiguous().view(-1, num_classes), labels.contiguous().view(-1))
        # Loss count
        total_loss += float(test_loss.data[0])
        total_images += float(images.size(0))
        batches_done += 1
        if debug_print is True:
            sys.stderr.write(str(confusion_matrix.conf)+"\n")
            sys.stderr.write(TextColor.BLUE+'Batches done: ' + str(batches_done) + " / " + str(len(validation_loader)) +
                             "\n" + TextColor.END)

    avg_loss = total_loss / total_images if total_images else 0

    sys.stderr.write(TextColor.YELLOW+'Test Loss: ' + str(avg_loss) + "\n"+TextColor.END)
    sys.stderr.write("Confusion Matrix \n: " + str(confusion_matrix.conf) + "\n" + TextColor.END)
    return {'loss': avg_loss}

