import argparse
import sys
import numpy as np
import torch
import torchnet.meter as meter
import torch.nn.parallel
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from modules.models.inception import Inception3

from modules.core.dataloader import PileupDataset, TextColor
np.set_printoptions(threshold=np.nan)
"""
This script will evaluate a model and output all the missed predictions in a CSV file.

Input:
- A trained model
- A test CSV file to evaluate

Output:
- A CSV file containing all the missed predictions
"""


def test(data_file, batch_size, model_path, gpu_mode, num_workers, num_classes=3):
    """
    Predict images of the data file and print names all the files that are missed in a CSV file.
    :param data_file: A CSV file containing image information
    :param batch_size: Size of the batch
    :param model_path: Path to a trained model
    :param gpu_mode: If true the model will use GPUs
    :param num_workers: Number of workers to use for loading data
    :param num_classes: Number of classes (HOM, HET, HOM_ALT)
    :return:
    """
    transformations = transforms.Compose([transforms.ToTensor()])

    # the validation dataset loader
    validation_data = PileupDataset(data_file, transformations)
    validation_loader = DataLoader(validation_data,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=num_workers,
                                   pin_memory=gpu_mode
                                   )
    sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)

    model = Inception3()

    if os.path.isfile(model_path) is False:
        sys.stderr.write(TextColor.RED + "ERROR: INVALID PATH TO THE MODEL\n")
        exit(1)
    sys.stderr.write(TextColor.GREEN + "INFO: MODEL LOADING\n" + TextColor.END)
    model = ModelHandler.load_model_for_training(model, model_path)
    sys.stderr.write(TextColor.GREEN + "INFO:  MODEL LOADED SUCCESSFULLY\n" + TextColor.END)

    if gpu_mode:
        model = model.cuda()

    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    if gpu_mode:
        model = model.cuda()

    # create a CSV file
    smry = open("test_miss_calls_" + data_file.split('/')[-1], 'w')

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Test the Model
    sys.stderr.write(TextColor.PURPLE + 'Test starting\n' + TextColor.END)
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

        # add to confusion matrix
        outputs = model(images)
        confusion_matrix.add(outputs.data.squeeze(), labels.data.type(torch.LongTensor))
        loss = criterion(outputs.contiguous().view(-1, num_classes), labels.contiguous().view(-1))
        # Loss count
        total_images += images.size(0)
        total_loss += loss.data[0]
        # converts predictions to numpy array
        preds_numpy = outputs.cpu().data.topk(1)[1].numpy().ravel().tolist()
        true_label_numpy = labels.cpu().data.numpy().ravel().tolist()

        eq = np.equal(preds_numpy, true_label_numpy)
        # find all mismatch indices
        mismatch_indices = np.where(eq == False)[0]

        # print all mismatch indices to the CSV file
        for index in mismatch_indices:
            smry.write(str(true_label_numpy[index]) + "," + str(preds_numpy[index]) + ","
                       + records[index] + "\n")

        batches_done += 1
        if batches_done % 50 == 0:
            sys.stderr.write(str(confusion_matrix.conf)+"\n")
            sys.stderr.write(TextColor.BLUE+'Batches done: ' + str(batches_done) + " / " + str(len(validation_loader)) +
                             "\n" + TextColor.END)

    # print summaries
    sys.stderr.write(TextColor.YELLOW+'Test Loss: ' + str(total_loss/total_images) + "\n"+TextColor.END)
    sys.stderr.write("Confusion Matrix \n: " + str(confusion_matrix.conf) + "\n" + TextColor.END)


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks.
    '''
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
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
        "--model_path",
        type=str,
        required=True,
        help="Path to a trained model"
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
        default=16,
        help="Batch size for training, default is 100."
    )

    FLAGS, unparsed = parser.parse_known_args()

    test(FLAGS.test_file, FLAGS.batch_size, FLAGS.model_path, FLAGS.gpu_mode, FLAGS.num_workers)
