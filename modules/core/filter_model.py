from modules.handlers.TsvWriter import TsvWriter
from modules.core.SplitDatasetLoader import SplitDataloader
import datetime
import sys
import os
from os import path
import numpy
import torch
import torch.nn as nn
import random
from matplotlib import pyplot
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import h5py


# results indices:
PERCENT_MATCHES, MATCHES, CONFUSION_VECTORS, N_FALSE_NEG, N_FALSE_POS, N_TRUE_NEG, N_TRUE_POS, SENSITIVITY, PRECISION = 0,1,2,3,4,5,6,7,8

POSITIVE_PREDICTION_THRESHOLD = 0.01

class ResultsHandler:
    def __init__(self, y_matrix, y_predict_matrix, x_matrix, metadata_matrix, dataset_train_length, dataset_test_length, loss_per_iteration, learning_rate, weight_decay, loss_fn, optimizer, model, header, output_directory="output/"):
        self.directory = output_directory
        self.y_matrix = y_matrix
        self.y_predict_matrix = y_predict_matrix
        self.x_matrix = x_matrix
        self.metadata_matrix = metadata_matrix
        self.header = header
        self.dataset_train_length = dataset_train_length
        self.dataset_test_length = dataset_test_length
        self.loss_per_iteration = loss_per_iteration
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn
        self.model = model
        self.optimizer = optimizer

        self.datetime_string = '-'.join(list(map(str, datetime.datetime.now().timetuple()))[:-1])
        self.directory = path.join(self.directory, 'run_' + self.datetime_string)
        self.tsv_writer = TsvWriter(output_directory=self.directory, filename_prefix="results", header=self.header)

        # ensure output directory exists
        if not path.exists(self.directory):
            os.mkdir(self.directory)

    def write_model_state_file(self):
        # save model file with unique datetime suffix
        torch.save(self.model.state_dict(), path.join(self.directory, "model"))

    def write_performance_stats(self):
        self.stats = calculate_testing_stats(y_matrix=self.y_matrix,
                                             y_predict_matrix=self.y_predict_matrix,
                                             x_matrix=self.x_matrix,
                                             metadata_matrix=self.metadata_matrix,
                                             output_directory=self.directory)

        results = [self.learning_rate, self.weight_decay, self.stats[N_FALSE_NEG], self.stats[N_FALSE_POS],
                   self.stats[N_TRUE_NEG], self.stats[N_TRUE_POS], self.stats[SENSITIVITY], self.stats[PRECISION]]

        self.tsv_writer.append_row(results)

    def save_full_output(self, y, y_predict_logits, metadata, output_directory):
        data = numpy.concatenate((y, y_predict_logits, metadata), axis=1)

        print(data.shape)

        filename = "truth_vs_prediction.npz"
        write_dataset(output_dir=output_directory,
                      filename=filename,
                      data=data)

    def write_full_output_dataset(self):
        self.save_full_output(y=self.y_matrix,
                              y_predict_logits=self.y_predict_matrix,
                              output_directory=self.directory,
                              metadata=self.metadata_matrix)

    def save_loss_plot(self, show=False):
        fig = pyplot.figure()
        fig.set_size_inches(10, 8)
        ax = pyplot.axes()
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        x_loss = list(range(len(self.loss_per_iteration)))
        pyplot.plot(x_loss, self.loss_per_iteration)

        if show:
            fig.show()

        t = datetime.datetime
        datetime_string = '-'.join(list(map(str, t.now().timetuple()))[:-1])
        fig.savefig(path.join(self.directory, "loss_" + datetime_string))
        pyplot.close()

    def write_parameter_info(self):
        with open(path.join(self.directory, "parameters"), 'w') as out_file:
            out_file.write("Train set size: " + str(self.dataset_train_length) + '\n')
            out_file.write("Test set size: " + str(self.dataset_test_length) + '\n')
            out_file.write("learning rate: " + str(self.learning_rate) + '\n')
            out_file.write("weight decay: " + str(self.weight_decay) + '\n')
            out_file.write("optimizer: " + str(self.optimizer) + '\n')
            out_file.write("loss function: " + str(self.loss_fn) + '\n')
            out_file.write("model: " + str(self.model) + '\n')

    def print_parameter_info(self):
        print("Train set size: ", self.dataset_train_length)
        print("Test set size: ", self.dataset_test_length)
        print("learning rate: ", self.learning_rate)
        print("weight decay: ", self.weight_decay)
        print("optimizer: ", self.optimizer)
        print("loss function: ", self.loss_fn)
        print("model: ", self.model)

    def print_performance_stats(self):
        print("Total sensitivity\t", float(self.stats[N_TRUE_POS]) / (self.stats[N_TRUE_POS] + self.stats[N_FALSE_NEG]))
        print("false negatives\t", self.stats[N_FALSE_NEG])
        print("false positives\t", self.stats[N_TRUE_NEG])
        print("true negatives\t", self.stats[N_FALSE_POS])
        print("true positives\t", self.stats[N_TRUE_POS])


class ShallowLinear(nn.Module):
    '''
    A simple model to take candidate site data vectors and predict their status as a variant or non-variant
    '''
    def __init__(self):
        super(ShallowLinear, self).__init__()

        self.layer_sizes = [27, 64, 1]
        D_in, H1, D_out = self.layer_sizes

        # print(D_in, H1, D_out)

        self.linear1 = nn.Linear(D_in, H1)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(H1, D_out)
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        # print(x.size())

        x = self.linear2(x)
        x = self.sigmoid1(x)
        # print(x.size())

        return x


class Linear(nn.Module):
    '''
    A simple model to take candidate site data vectors and predict their status as a variant or non-variant
    '''
    def __init__(self):
        super(Linear, self).__init__()

        self.layer_sizes = [27, 1]
        D_in, D_out = self.layer_sizes

        # print(D_in, D_out)

        self.linear1 = nn.Linear(D_in, D_out)
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid1(x)
        # print(x.size())

        return x


class CandidateAlleleDataset(Dataset):
    def __init__(self, data):
        # check if path exists
        # print(data.shape)

        x = data[:,6:-1]
        y = data[:,-1:]
        metadata = data[:,:5]
        # data_vcf_filter = data[,5:6]

        x_dtype = torch.FloatTensor
        y_dtype = torch.FloatTensor     # for MSE Loss or BCE loss
        # y_dtype = torch.LongTensor      # for CE Loss

        self.length = x.shape[0]

        self.x_data = torch.from_numpy(x).type(x_dtype)
        self.y_data = torch.from_numpy(y).type(y_dtype)

        self.metadata = metadata

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.metadata[index]

    def __len__(self):
        return self.length


class CandidateAlleleH5Dataset(Dataset):
    """
    If dataset is too large to load into memory, and exists in h5py format, use this
    A list of indices defining the subset of the full dataset must be provided.
    """
    def __init__(self, h5_dataset, indices):
        # check if path exists
        self.data = h5_dataset
        self.indices = indices
        self.length = len(indices)

        self.x_dtype = torch.FloatTensor
        self.y_dtype = torch.FloatTensor     # for MSE Loss or BCE loss
        # self.y_dtype = torch.LongTensor      # for CE Loss

    def __getitem__(self, index):
        index = self.indices[index]

        x = self.data[index, 6:-1]
        y = self.data[index, -1:]
        metadata = self.data[index, :5]

        # print(x.shape, y.shape, metadata.shape)

        x_data = torch.from_numpy(x).type(self.x_dtype)
        y_data = torch.from_numpy(y).type(self.y_dtype)

        return x_data, y_data, metadata

    def __len__(self):
        return self.length


class CandidateAlleleNumpyTestDataset(Dataset):
    """
    If dataset is too large to load into memory, and exists in h5py format, use this
    A list of indices defining the subset of the full dataset must be provided.
    """
    def __init__(self, numpy_dataset, transform=None):
        # check if path exists
        self.data = numpy_dataset.T
        self.transform = None

        # print("before transform:", type(self.data))
        # print(self.data.shape)

        self.length = self.data.shape[0]

        self.x_dtype = torch.FloatTensor
        self.y_dtype = torch.FloatTensor     # for MSE Loss or BCE loss
        # self.y_dtype = torch.LongTensor      # for CE Loss

    def __getitem__(self, index):
        x = self.data[index, 2:]
        coordinates = self.data[index, :2]

        # print("dataloader x:",x.shape)
        # print(index)
        # print(x)

        x_data = torch.from_numpy(x).type(self.x_dtype)

        # if index == 0:
        #     print("after transform data:", type(self.data))
        #     print("after transform coords:", type(coordinates))

        return x_data, coordinates

    def __len__(self):
        return self.length


def subset_coordinates_by_positive_prediction(coordinates_matrix, y_predict_matrix):
    # print("x shape:", coordinates_matrix.shape)

    coordinates = coordinates_matrix[:,:2]

    # print("coordinates shape:", coordinates.shape)

    positive_mask = (y_predict_matrix > POSITIVE_PREDICTION_THRESHOLD).squeeze()

    # print("positives:", numpy.count_nonzero(positive_mask))

    coordinates_subset = coordinates[positive_mask, :]

    numpy.savetxt('test.out', coordinates_subset, delimiter=',')

    return coordinates_subset


def save_full_output(y, y_predict_logits, metadata, output_directory):
    if not path.exists(output_directory):
        os.mkdir(output_directory)

    data = numpy.concatenate((y, y_predict_logits, metadata), axis=1)

    # print(data.shape)

    filename = "truth_vs_prediction.npz"
    write_dataset(output_dir=output_directory,
                  filename=filename,
                  data=data)


def write_dataset(output_dir, filename, data):
    """
    Create a npz training set of all labeled candidate sites found in the region
    :return:
    """
    #filename = "candidate_frequencies_confusion.npz"

    numpy.savez_compressed(path.join(output_dir, filename), a=data)


def predict(model, loader):
    coordinate_vectors = list()
    x_vectors = list()
    y_predict_vectors = list()

    batch_index = 0
    for x, coordinates in loader:
        x = Variable(x)

        y_predict = model.forward(x)

        coordinate_vectors.append(coordinates.numpy())
        x_vectors.append(x.data.numpy())
        y_predict_vectors.append(y_predict.data.numpy())

        batch_index += 1

    y_predict_matrix = numpy.concatenate(y_predict_vectors)
    x_matrix = numpy.concatenate(x_vectors)

    coordinates = numpy.concatenate(coordinate_vectors)
    # print("after dataloader:", type(coordinate_vectors[0]))
    #
    # except ValueError:
    #     print(len(y_predict_vectors))
    #     print(len(x_vectors))
    #     print(len(coordinate_vectors))
    #
    #     for data in zip(y_predict_vectors,x_vectors,coordinate_vectors):
    #         print(data[0][0])
    #         print(data[1][0])
    #         print(data[2][0])

    return y_predict_matrix, x_matrix, coordinates


def train_batch(model, x, y, optimizer, loss_fn):
    # Run forward calculation
    y_predict = model.forward(x)

    # Compute loss.
    loss = loss_fn(y_predict, y)

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable weights
    # of the model)
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

    return loss.data[0]


def train(model, loader, optimizer, loss_fn, epochs=5, cutoff=None, print_progress=False):
    losses = list()

    batch_index = 0
    i = 0
    for e in range(epochs):
        for x, y, metadata in loader:
            x = Variable(x)
            y = Variable(y)

            loss = train_batch(model=model, x=x, y=y, optimizer=optimizer, loss_fn=loss_fn)
            losses.append(loss)

            batch_index += 1

            if print_progress and i % 100 == 0:
                print(i, loss)

            i += 1

            if cutoff is not None and i > cutoff:
                break
        if cutoff is not None and i > cutoff:
            break

        if print_progress:
            print("Epoch: ", e+1)
            print("Batches: ", batch_index)

    return losses


def test_batch(model, x, y):
    # run forward calculation
    y_predict = model.forward(x)

    return y, y_predict


def test(model, loader):
    metadata_vectors = list()
    x_vectors = list()
    y_vectors = list()
    y_predict_vectors = list()

    batch_index = 0
    for x, y, metadata in loader:
        x = Variable(x)
        y = Variable(y)

        y, y_predict = test_batch(model=model, x=x, y=y)

        metadata_vectors.append(metadata)
        x_vectors.append(x.data.numpy())
        y_vectors.append(y.data.numpy())
        y_predict_vectors.append(y_predict.data.numpy())

        batch_index += 1

    y_predict_matrix = numpy.concatenate(y_predict_vectors)
    y_matrix = numpy.concatenate(y_vectors)
    x_matrix = numpy.concatenate(x_vectors)
    metadata = numpy.concatenate(metadata_vectors)

    return y_predict_matrix, y_matrix, x_matrix, metadata


def assess_prediction(y, y_predict):
    # print(y[:1, :9])
    # print(y_predict[:1, :9])

    y_argmax = numpy.argmax(y, axis=1)
    y_predict_argmax = numpy.argmax(y_predict, axis=1)

    # print(y_argmax[:1])
    # print(y_predict_argmax[:1])

    confusion = (y_argmax != y_predict_argmax).squeeze()
    n_confusion = sum(confusion)

    accuracy = 1-(n_confusion/y.shape[0])

    print("Accuracy: ", accuracy)

    pyplot.hist(y_predict_argmax, bins=list(range(46)))
    pyplot.show()
    pyplot.hist(y_argmax, bins=list(range(46)))
    pyplot.show()


def calculate_testing_stats(y_matrix, y_predict_matrix, x_matrix, metadata_matrix, output_directory, filter_false_positive=True):
    # convert labels to flattened 1d vector
    truth_labels = numpy.squeeze(y_matrix)
    predict_labels = numpy.squeeze(y_predict_matrix).round()  #assume 0.5 threshold

    # print(predict_labels[0:4])
    # print(predict_labels.round()[0:4])
    # print(truth_labels.shape, predict_labels.shape, x_matrix.shape)

    # get boolean mask vectors for entries that have correct vs incorrect predictions
    equivalency_mask_vector = (predict_labels == truth_labels)
    non_equivalency_mask_vector = numpy.invert(equivalency_mask_vector)

    # get vectors that had incorrect classifications
    confusion_vectors = x_matrix[non_equivalency_mask_vector,:]
    confusion_labels = truth_labels[non_equivalency_mask_vector]

    # get vectors that had correct classifications
    non_confusion_vectors = x_matrix[equivalency_mask_vector,:]
    non_confusion_labels = truth_labels[equivalency_mask_vector]

    # get boolean mask vectors for labels that are 1 or 0
    positive_mask_confusion = (confusion_labels == 1)
    negative_mask_confusion = (confusion_labels == 0)

    positive_mask_nonconfusion = (non_confusion_labels == 1)
    negative_mask_nonconfusion = (non_confusion_labels == 0)

    # get false positives and false negatives
    false_negative = confusion_labels[positive_mask_confusion]
    false_positive = confusion_labels[negative_mask_confusion]
    true_negative = non_confusion_labels[negative_mask_nonconfusion]
    true_positive = non_confusion_labels[positive_mask_nonconfusion]

    n_false_negative = len(false_negative)
    n_false_positive = len(false_positive)
    n_true_negative = len(true_negative)
    n_true_positive = len(true_positive)

    length = len(confusion_labels)
    confusion_labels = confusion_labels.reshape((length, 1))

    # print(confusion_vectors.shape, confusion_labels.shape)

    labeled_confusion_vectors = numpy.concatenate((confusion_vectors, confusion_labels), axis=1)

    if filter_false_positive:
        labeled_confusion_vectors = labeled_confusion_vectors[numpy.squeeze(positive_mask_confusion),:]

    total_matches = numpy.count_nonzero(equivalency_mask_vector)
    percent_matches = float(total_matches)/len(truth_labels)

    sensitivity = float(n_true_positive) / (n_true_positive + n_false_negative)
    precision = float(n_true_positive) / (n_true_positive + n_false_positive)

    return percent_matches, total_matches, labeled_confusion_vectors, n_false_negative, n_false_positive, n_true_negative, n_true_positive, sensitivity, precision


def grid_search(dataset_train, dataset_test, tsv_writer):
    # Batch size is the number of training examples used to calculate each iteration's gradient
    batch_size_train = 128
    n_epochs = 3

    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size_train, shuffle=True)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=len(dataset_test), shuffle=False)

    a = numpy.arange(2, 6.25, step=0.25)
    parameter_range = [10 ** (-x) for x in a]

    n_repetitions = 3

    i = 0
    for learning_rate in parameter_range:
        for weight_decay in parameter_range:
            for r in range(n_repetitions):
                # Instantiate model
                model = ShallowLinear()

                # Initialize the optimizer with above parameters
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

                # Define the loss function
                loss_fn = nn.BCELoss()  # mean squared error

                # Train and get the resulting loss per iteration
                loss_per_iteration = train(model=model, loader=data_loader_train, optimizer=optimizer, loss_fn=loss_fn,
                                           epochs=n_epochs, cutoff=None)

                # Test and get the resulting predicted y values
                y_predict_matrix, y_matrix, x_matrix, metadata_matrix = test(model=model, loader=data_loader_test)

                stats = calculate_testing_stats(y_matrix=y_matrix,
                                                y_predict_matrix=y_predict_matrix,
                                                x_matrix=x_matrix,
                                                metadata_matrix=metadata_matrix,
                                                output_directory=None)

                results = [learning_rate, weight_decay, stats[N_FALSE_NEG], stats[N_FALSE_POS], stats[N_TRUE_NEG], stats[N_TRUE_POS], stats[SENSITIVITY], stats[PRECISION]]
                tsv_writer.append_row(results)

                if i == 0:
                    print("optimizer: ", optimizer)
                    print("loss function: ", loss_fn)
                    print("model: ", model)
                    print("batch size: ", batch_size_train)
                    print("epochs: ", n_epochs)

                print(i, r, learning_rate, weight_decay)

                i += 1

def run_prediction(test_dataset, model_state_path=None):
    # Batch size is the number of training examples used to calculate each iteration's gradient
    batch_size_test = 8192

    # Instantiate model
    model = ShallowLinear()

    # Initialize data loader
    data_loader_test = DataLoader(dataset=test_dataset, batch_size=batch_size_test, )

    # load previous model and test only
    model.load_state_dict(torch.load(model_state_path))

    # Test and get the resulting predicted y values
    y_predict_matrix, x_matrix, coordinates = predict(model=model, loader=data_loader_test)

    return y_predict_matrix, x_matrix, coordinates


def run(train_paths, test_paths, train_length, test_length, load_model=False, model_state_path=None):
    # Batch size is the number of training examples used to calculate each iteration's gradient
    batch_size_train = 128
    batch_size_test = 4096
    n_epochs = 1

    cutoff = None
    downsample = False

    # Define the hyperparameters
    learning_rate = 0.005623413251903491
    weight_decay = 1.778279410038923e-06

    # Instantiate model
    model = ShallowLinear()

    # Initialize the optimizer with above parameters
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Define the loss function
    loss_fn = nn.BCELoss()      # binary cross entropy loss

    # Initialize data loader
    data_loader_test = SplitDataloader(file_paths=test_paths, length=test_length, batch_size=batch_size_test)

    if not load_model:
        # initialize training set data loader
        data_loader_train = SplitDataloader(file_paths=train_paths, length=train_length, batch_size=batch_size_train,
                                            downsample=downsample)

        # Train and get the resulting loss per iteration
        loss_per_iteration = train(model=model, loader=data_loader_train, optimizer=optimizer, loss_fn=loss_fn,
                                   epochs=n_epochs, cutoff=cutoff, print_progress=True)
    else:
        # load previous model and test only
        model.load_state_dict(torch.load(model_state_path))
        loss_per_iteration = list()

    # Test and get the resulting predicted y values
    y_predict_matrix, y_matrix, x_matrix, metadata = test(model=model, loader=data_loader_test)

    return loss_per_iteration, y_predict_matrix, y_matrix, x_matrix, metadata, learning_rate, weight_decay, optimizer, loss_fn, model


def predict_sites(model_state_path, numpy_dataset):
    test_dataset = CandidateAlleleNumpyTestDataset(numpy_dataset=numpy_dataset)

    # print("Test set size: ", test_dataset.length)

    y_predict_matrix, \
    x_matrix, \
    coordinates_matrix = run_prediction(test_dataset=test_dataset,
                                        model_state_path=model_state_path)

    predicted_coordinates = subset_coordinates_by_positive_prediction(coordinates_matrix=coordinates_matrix,
                                                                      y_predict_matrix=y_predict_matrix)

    # data = numpy.concatenate([coordinates_matrix, x_matrix, y_predict_matrix], axis=1)
    # numpy.savetxt('test.out', data, delimiter=',')

    return predicted_coordinates


def main():
    load_model = False

    model_state_path = "/home/ryan/code/variant_classifier/output/WG_GIAB_0_threshold_run_2018-4-27-10-36-23-4-117/model"
    output_directory = "output/"

    dataset_log_path = "/home/ryan/data/GIAB/filter_model_training_data/vision/WG/0_threshold/confident/chr1_19__0_all_1_coverage/dataset_log.tsv"

    # training_set_relative_size = 0.7
    # train_paths, test_paths, train_length, test_length = SplitDataloader.partition_dataset_paths(dataset_log_path=dataset_log_path, train_size_proportion=training_set_relative_size)
    train_paths, test_paths, train_length, test_length = SplitDataloader.partition_dataset_paths_by_chromosome(dataset_log_path=dataset_log_path, test_chromosome_name_list=['19'])

    # for path in sorted(test_paths):
    #     print(path)
    # print(len(test_paths))
    # exit()

    print("Train set size: ", train_length)
    print("Test set size: ", test_length)

    header = ["learning rate", "weight decay", "false negative", "false positive", "true negative", "true positive",
              "sensitivity", "precision"]

    loss_per_iteration, \
    y_predict_matrix, \
    y_matrix, \
    x_matrix, \
    metadata_matrix, \
    learning_rate, \
    weight_decay, \
    optimizer, \
    loss_fn, \
    model = run(train_paths=train_paths,
                test_paths=test_paths,
                train_length=train_length,
                test_length=test_length,
                load_model=load_model,
                model_state_path=model_state_path)

    results_handler = ResultsHandler(y_matrix=y_matrix,
                                     y_predict_matrix=y_predict_matrix,
                                     x_matrix=x_matrix,
                                     metadata_matrix=metadata_matrix,
                                     dataset_train_length=train_length,
                                     dataset_test_length=test_length,
                                     loss_per_iteration=loss_per_iteration,
                                     learning_rate=learning_rate,
                                     weight_decay=weight_decay,
                                     loss_fn=loss_fn,
                                     optimizer=optimizer,
                                     model=model,
                                     header=header,
                                     output_directory=output_directory)

    results_handler.write_performance_stats()
    results_handler.write_model_state_file()
    results_handler.write_parameter_info()
    results_handler.write_full_output_dataset()
    results_handler.save_loss_plot()

    results_handler.print_parameter_info()
    results_handler.print_performance_stats()


if __name__ == "__main__":
    main()
