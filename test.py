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
from tqdm import tqdm
from modules.models.ModelHandler import ModelHandler
from modules.models.resnet import resnet18_custom

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
    model = ModelHandler.get_new_model(gpu_mode)

    if os.path.isfile(model_path) is False:
        sys.stderr.write(TextColor.RED + "ERROR: INVALID PATH TO RETRAIN PATH MODEL --retrain_model_path\n")
        exit(1)
    model = ModelHandler.load_model(model, model_path)
    sys.stderr.write(TextColor.GREEN + "INFO: MODEL LOADED\n" + TextColor.END)

    if gpu_mode:
        model = torch.nn.DataParallel(model).cuda()

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Test the Model
    total_loss = 0
    total_images = 0
    accuracy = 0
    confusion_matrix = meter.ConfusionMeter(num_classes)

    # create a CSV file
    smry = open("test_miss_calls_" + data_file.split('/')[-1], 'w')

    with torch.no_grad():
        with tqdm(total=len(validation_loader), desc='Accuracy: ', ncols=100) as pbar:
            for i, (images, labels, records) in enumerate(validation_loader):
                if gpu_mode:
                    images = images.cuda()
                    labels = labels.cuda()

                # Predict + confusion_matrix + loss
                predictions = model(images)
                confusion_matrix.add(predictions.data, labels.data)

                m = nn.Softmax(dim=1)
                soft_probs = m(predictions)
                preds = soft_probs.cpu()

                # converts predictions to numpy array
                preds_numpy = preds.cpu().data.topk(1)[1].numpy().ravel().tolist()
                true_label_numpy = labels.cpu().data.numpy().ravel().tolist()

                eq = np.equal(preds_numpy, true_label_numpy)
                # find all mismatch indices
                mismatch_indices = np.where(eq == False)[0]

                # print all mismatch indices to the CSV file
                for index in mismatch_indices:
                    smry.write(str(true_label_numpy[index]) + "\t" + str(preds_numpy[index]) + "\t"
                               + records[index] + "\t" + str(preds[index]) + "\n")

                loss = criterion(predictions.contiguous().view(-1, num_classes), labels.contiguous().view(-1))

                # Progress bar update
                pbar.update(1)
                cm_value = confusion_matrix.value()
                denom = (cm_value.sum() - cm_value[0][0]) if (cm_value.sum() - cm_value[0][0]) > 0 else 1.0
                accuracy = 100.0 * (cm_value[1][1] + cm_value[2][2]) / denom
                pbar.set_description("Accuracy: " + str(accuracy))

                # loss count
                total_loss += loss.item()
                total_images += (images.size(0))

    avg_loss = total_loss / total_images if total_images else 0
    sys.stderr.write(TextColor.BLUE + 'Test Loss: ' + str(avg_loss) + "\n" + TextColor.END)
    sys.stderr.write(TextColor.GREEN + "Confusion Matrix: \n" + str(confusion_matrix.conf) + "\n" + TextColor.END)


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks.
    '''
    parser = argparse.ArgumentParser()
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