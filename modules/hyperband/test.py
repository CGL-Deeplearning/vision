from __future__ import print_function
import sys
import torch
import torchnet.meter as meter
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from modules.core.dataloader import PileupDataset, TextColor
"""
This script will evaluate a model and return the loss value.

Input:
- A trained model
- A test CSV file to evaluate

Returns:
- Loss value
"""


def test(test_file, batch_size, gpu_mode, trained_model, num_workers, num_classes):
    """
    Test a trained model
    :param test_file: Test CSV file containing the test set
    :param batch_size: Batch size for prediction
    :param gpu_mode: If True GPU will be used
    :param trained_model: Trained model
    :param num_workers: Number of workers for data loader
    :param num_classes: Number of output classes (3- HOM, HET, HOM_ALT)
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
    sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)

    # set the evaluation mode of the model
    test_model = trained_model.eval()
    if gpu_mode:
        test_model = test_model.cuda()

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Test the Model
    sys.stderr.write(TextColor.PURPLE + 'Test starting\n' + TextColor.END)
    total_loss = 0
    total_images = 0
    accuracy = 0
    confusion_matrix = meter.ConfusionMeter(num_classes)
    with torch.no_grad():
        with tqdm(total=len(validation_loader), desc='Accuracy: ', leave=True, dynamic_ncols=True) as pbar:
            for i, (images, labels, records) in enumerate(validation_loader):
                if gpu_mode:
                    images = images.cuda()
                    labels = labels.cuda()

                # Predict + confusion_matrix + loss
                outputs = test_model(images)
                confusion_matrix.add(outputs.data, labels.data)

                loss = criterion(outputs.contiguous().view(-1, num_classes), labels.contiguous().view(-1))

                # loss count
                total_loss += loss.item()
                total_images += (images.size(0))

                # Progress bar update
                pbar.update(1)
                cm_value = confusion_matrix.value()
                denom = (cm_value.sum() - cm_value[0][0]) if (cm_value.sum() - cm_value[0][0]) > 0 else 1.0
                accuracy = 100.0 * (cm_value[1][1] + cm_value[2][2]) / denom
                pbar.set_description("Accuracy: " + str(accuracy))

    avg_loss = total_loss / total_images if total_images else 0

    sys.stderr.write(TextColor.YELLOW + 'Test Loss: ' + str(avg_loss) + "\n" + TextColor.END)
    sys.stderr.write("Confusion Matrix: \n" + str(confusion_matrix.conf) + "\n" + TextColor.END)

    return {'loss': avg_loss, 'accuracy': accuracy}

