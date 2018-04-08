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

def test(data_file, batch_size, gpu_mode, trained_model, num_classes, num_workers):
    transformations = transforms.Compose([transforms.ToTensor()])

    validation_data = PileupDataset(data_file, transformations)
    validation_loader = DataLoader(validation_data,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=num_workers,
                                   pin_memory=gpu_mode
                                   )
    sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)

    test_model = trained_model.eval()
    if gpu_mode:
        test_model = test_model.cuda()

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

        # Forward + Backward + Optimize
        outputs = test_model(images)
        confusion_matrix.add(outputs.data, labels.data)
        loss = criterion(outputs.contiguous().view(-1, num_classes), labels.contiguous().view(-1))
        # Loss count
        total_images += images.size(0)
        total_loss += loss.data[0]

        batches_done += 1
        if batches_done % 100 == 0:
            sys.stderr.write(str(confusion_matrix.conf)+"\n")
            sys.stderr.write(TextColor.BLUE+'Batches done: ' + str(batches_done) + " / " + str(len(validation_loader)) +
                             "\n" + TextColor.END)

    print('Test Loss: ' + str(total_loss/total_images))
    # print('Confusion Matrix: \n', confusion_matrix.conf)

    sys.stderr.write(TextColor.YELLOW+'Test Loss: ' + str(total_loss/total_images) + "\n"+TextColor.END)
    sys.stderr.write("Confusion Matrix \n: " + str(confusion_matrix.conf) + "\n" + TextColor.END)

    return total_loss / total_images if total_images else 0


def train(train_file, validation_file, batch_size, epoch_limit, file_name, gpu_mode, num_workers, retrain_mode,
          model_path, num_classes=3):
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

    if retrain_mode is False:
        model = Inception3()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-6, nesterov=True)
        if gpu_mode:
            model = model.cuda()
    else:
        model, optimizer = get_model_and_optimizer(retrain_mode, model_path, gpu_mode)

    # Loss function
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0

    if gpu_mode:
        model = torch.nn.DataParallel(model).cuda()

    running_test_loss = -1

    # Train the Model
    sys.stderr.write(TextColor.PURPLE + 'Training starting\n' + TextColor.END)
    for epoch in range(start_epoch, epoch_limit, 1):
        total_loss = 0
        total_images = 0
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
            total_images += (x.size(0))
            total_loss += loss.data[0]
            batches_done += 1

            if batches_done % 10 == 0:
                avg_loss = total_loss / total_images if total_images else 0
                print(str(epoch + 1) + "\t" + str(i + 1) + "\t" + str(avg_loss))
                sys.stderr.write(TextColor.BLUE + "EPOCH: " + str(epoch+1) + " Batches done: " + str(batches_done)
                                 + " / " + str(len(train_loader)) + "\n" + TextColor.END)
                sys.stderr.write(TextColor.YELLOW + " Training loss: " + str(avg_loss) + "\n" + TextColor.END)
                sys.stderr.write(TextColor.DARKCYAN + "Time Elapsed: " + str(time.time() - start_time) +
                                 "\n" + TextColor.END)
                start_time = time.time()

            # if batches_done % 8 == 0:
            #     current_test_loss = test(validation_file, batch_size, gpu_mode, model, num_classes, num_workers)
            #     if running_test_loss == -1 or current_test_loss < running_test_loss:
            #         save_best_model(model, optimizer, current_test_loss, file_name)
            #         running_test_loss = current_test_loss

        avg_loss = total_loss/total_images if total_images else 0
        sys.stderr.write(TextColor.BLUE + "EPOCH: " + str(epoch+1) + " Completed: " + str(i+1) + "\n" + TextColor.END)
        sys.stderr.write(TextColor.YELLOW + " Training loss: " + str(avg_loss) + "\n" + TextColor.END)

        print(str(epoch+1) + "\t" + str(i + 1) + "\t" + str(avg_loss))

        avg_loss = total_loss / total_images if total_images else 0
        sys.stderr.write(TextColor.YELLOW + 'EPOCH: ' + str(epoch+1))
        sys.stderr.write(' Train Loss: ' + str(avg_loss) + "\n" + TextColor.END)

        # After each epoch do validation
        current_test_loss = test(validation_file, batch_size, gpu_mode, model, num_classes, num_workers)
        if running_test_loss == -1 or current_test_loss < running_test_loss:
            save_best_model(model, optimizer, current_test_loss, file_name)
            running_test_loss = current_test_loss

    sys.stderr.write(TextColor.PURPLE + 'Finished training\n' + TextColor.END)


def save_best_model(best_model, optimizer, current_loss, file_name):
    sys.stderr.write(TextColor.BLUE + "SAVING MODEL" + TextColor.END)
    if os.path.isfile(file_name + '_model.pkl'):
        os.remove(file_name + '_model.pkl')
    if os.path.isfile(file_name + '_checkpoint.pkl'):
        os.remove(file_name + '_checkpoint.pkl')
    torch.save(best_model, file_name + '_model.pkl')
    ModelHandler.save_checkpoint({
        'state_dict': best_model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, file_name + '_checkpoint.pkl')
    sys.stderr.write(TextColor.RED + " MODEL SAVED. LOSS: " + str(current_loss) + "\n" + TextColor.END)


def directory_control(file_path):
    directory = os.path.dirname(file_path)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)


def get_model_and_optimizer(model_retrain, model_checkpoint_path, gpu_mode):
    if model_retrain is True:
        model = ModelHandler.load_model_for_training(model_checkpoint_path, gpu_mode)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
        optimizer = ModelHandler.load_optimizer(optimizer, model_checkpoint_path, gpu_mode)
        return model, optimizer
    else:
        model = ModelHandler.get_new_model(gpu_mode)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
        return model, optimizer


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks to generate the pileup.
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
        "--validation_file",
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
        "--model_path",
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
    training_model, training_optimizer = get_model_and_optimizer(FLAGS.retrain_model, FLAGS.model_path, FLAGS.gpu_mode)
    directory_control(FLAGS.model_out.rpartition('/')[0]+"/")
    train(FLAGS.train_file, FLAGS.validation_file, FLAGS.batch_size, FLAGS.epoch_size, FLAGS.model_out, FLAGS.gpu_mode,
          FLAGS.num_workers, FLAGS.retrain_model, FLAGS.model_path)


