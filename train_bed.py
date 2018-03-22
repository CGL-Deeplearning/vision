import sys
import time
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torchnet.meter as meter

from modules.core.dataloader_bed import DataSetLoader
from modules.handlers.TextColor import TextColor
from modules.models.inception import Inception3
import numpy as np
np.set_printoptions(threshold=np.nan)

NUM_CLASSES = 3


def handle_directory(directory_path):
    """
    Create a directory if doesn't exist
    :param directory_path: path to the directory
    :return: desired directory name
    """
    # if directory has no trailing '/' then add it
    if directory_path[-1] != '/':
        directory_path += '/'
    # if directory doesn't exist then create it
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

    return directory_path


def holdout_test(bam_file, ref_file, holdout_test_file, batch_size, gpu_mode, trained_model, max_threads):
    transformations = transforms.Compose([transforms.ToTensor()])

    validation_data = DataSetLoader(bam_file, ref_file, holdout_test_file, transformations)
    validation_loader = DataLoader(validation_data,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=max_threads,
                                   pin_memory=gpu_mode
                                   )
    sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)

    model = trained_model.eval()
    if gpu_mode:
        model = model.cuda()

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Test the Model
    sys.stderr.write(TextColor.PURPLE + 'Test starting\n' + TextColor.END)
    total_loss = 0
    total_images = 0
    batches_done = 0
    confusion_matrix = meter.ConfusionMeter(NUM_CLASSES)
    for i, (images, labels, bed_record, summary_string) in enumerate(validation_loader):
        if gpu_mode is True and images.size(0) % 8 != 0:
            continue

        images = Variable(images, volatile=True)
        labels = Variable(labels, volatile=True)
        if gpu_mode:
            images = images.cuda()
            labels = labels.cuda()

        # Forward + Backward + Optimize
        outputs = model(images)
        confusion_matrix.add(outputs.data.squeeze(), labels.data.type(torch.LongTensor))
        loss = criterion(outputs.contiguous().view(-1, NUM_CLASSES), labels.contiguous().view(-1))
        # Loss count
        total_images += images.size(0)
        total_loss += loss.data[0]

        batches_done += 1
        if batches_done % 10 == 0:
            sys.stderr.write(str(confusion_matrix.conf)+"\n")
            sys.stderr.write(TextColor.BLUE + 'Batches done: ' + str(batches_done) +
                             " / " + str(len(validation_loader)) + "\n" + TextColor.END)

    print('Test Loss: ' + str(total_loss/total_images))
    print('Confusion Matrix: \n', confusion_matrix.conf)

    sys.stderr.write(TextColor.YELLOW+'Test Loss: ' + str(total_loss/total_images) + "\n"+TextColor.END)
    sys.stderr.write("Confusion Matrix \n: " + str(confusion_matrix.conf) + "\n" + TextColor.END)


def save_checkpoint(state, filename):
    torch.save(state, filename)


def save_model_checkpoint(model, output_dir, epoch, optimizer, batch):
    torch.save(model, output_dir + 'checkpoint_' + str(epoch + 1) + '_model.pkl')
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, output_dir + 'checkpoint_' + str(epoch + 1) + "." + str(batch + 1) + "_params.pkl")


def train(bam_file, ref_file, train_bed, val_bed, batch_size, epoch_limit, output_dir, gpu_mode, max_threads):

    transformations = transforms.Compose([transforms.ToTensor()])
    train_data_set = DataSetLoader(bam_file, ref_file, train_bed, transformations)
    train_loader = DataLoader(train_data_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=max_threads,
                              pin_memory=gpu_mode
                              )
    sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)
    model = Inception3()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    start_epoch = 0

    if gpu_mode:
        model = torch.nn.DataParallel(model).cuda()

    # Train the Model
    sys.stderr.write(TextColor.PURPLE + 'Training starting\n' + TextColor.END)
    for epoch in range(start_epoch, epoch_limit, 1):
        total_loss = 0
        total_images = 0
        batches_done = 0
        absolute_start = time.time()
        start_time = time.time()
        debug_st_time = time.time()
        for i, (images, labels, bed_record) in enumerate(train_loader):
            if gpu_mode is True and images.size(0) % 8 != 0:
                continue

            # THIS BLOCK IS FOR TESTING #
            # print(bed_record[0].replace('\t',' '))
            # analyze_np_array(images[0])
            # print('DATA LOADING TIME: ', i, (time.time()-start_time))
            # start_time = time.time()
            # start_time = time.time()
            # exit()
            # DO NOT REMOVE # - KISHWAR

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
            loss = criterion(outputs.contiguous().view(-1, NUM_CLASSES), y.contiguous().view(-1))
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
                sys.stderr.write(TextColor.YELLOW + " Loss: " + str(avg_loss) + "\n" + TextColor.END)
                sys.stderr.write(TextColor.DARKCYAN + "Time Elapsed: " + str((time.time() - start_time)) +
                                 " Secs \n" + TextColor.END)
                start_time = time.time()

        avg_loss = total_loss/total_images if total_images else 0
        sys.stderr.write(TextColor.BLUE + "EPOCH: " + str(epoch+1)
                         + " Batches done: " + str(i+1) + "/" + str(len(train_loader)) + "\n" + TextColor.END)
        sys.stderr.write(TextColor.YELLOW + " Loss: " + str(avg_loss) + "\n" + TextColor.END)
        sys.stderr.write(TextColor.DARKCYAN + "Time Elapsed: " + str((time.time() - absolute_start)) +
                         " Secs \n" + TextColor.END)
        print(str(epoch+1) + "\t" + str(i + 1) + "\t" + str(avg_loss))

        if (i+1) % 1000 == 0:
            save_model_checkpoint(model, output_dir, epoch, optimizer, i+1)
            sys.stderr.write(TextColor.RED+" MODEL SAVED \n" + TextColor.END)

        avg_loss = total_loss / total_images if total_images else 0
        sys.stderr.write(TextColor.YELLOW + 'EPOCH: ' + str(epoch+1))
        sys.stderr.write(' Loss: ' + str(avg_loss) + "\n" + TextColor.END)

        save_model_checkpoint(model, output_dir, epoch, optimizer, i + 1)

        # After each epoch do validation
        holdout_test(bam_file, ref_file, val_bed, batch_size, gpu_mode, model, max_threads)

    sys.stderr.write(TextColor.PURPLE + 'Finished training\n' + TextColor.END)

    torch.save(model, output_dir+'final_model.pkl')
    save_checkpoint({
        'epoch': epoch_limit,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, output_dir + 'final_params.pkl')
    sys.stderr.write(TextColor.PURPLE + 'Model saved as:' + output_dir + '_final.pkl\n' + TextColor.END)
    sys.stderr.write(TextColor.PURPLE + 'Model parameters saved as:' + output_dir + '_final_params.pkl\n' + TextColor.END)


def directory_control(file_path):
    if file_path[-1] != "/":
        file_path += "/"

    directory = os.path.dirname(file_path)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)
    return file_path


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks to generate the pileup.
    '''
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--bam",
        type=str,
        required=True,
        help="BAM file containing reads of interest."
    )
    parser.add_argument(
        "--ref",
        type=str,
        required=True,
        help="Reference corresponding to the BAM file."
    )
    parser.add_argument(
        "--train_bed",
        type=str,
        required=True,
        help="bed file path."
    )
    parser.add_argument(
        "--holdout_bed",
        type=str,
        required=True,
        help="Training data description csv file."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=20,
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
        "--img_output_dir",
        type=str,
        default="output/",
        help="Name of output directory to save images"
    )
    parser.add_argument(
        "--gpu_mode",
        type=bool,
        default=False,
        help="If true then cuda is on."
    )
    parser.add_argument(
        "--max_threads",
        type=int,
        required=False,
        default=80,
        help="Maximum number of threads to use when loading data."
    )
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.img_output_dir = handle_directory(FLAGS.img_output_dir)

    FLAGS.model_out = directory_control(FLAGS.model_out)
    train(FLAGS.bam, FLAGS.ref, FLAGS.train_bed, FLAGS.holdout_bed, FLAGS.batch_size,
          FLAGS.epoch_size, FLAGS.model_out, FLAGS.gpu_mode, FLAGS.max_threads)
