from modules.handlers.FileManager import FileManager
import os
import numpy
import pandas
import random


class SplitDataloader:
    def __init__(self, file_paths, length, batch_size, desired_n_to_p_ratio=20, downsample=False):
        self.file_paths = file_paths
        self.path_iterator = iter(file_paths)
        self.length = length
        self.n_files = len(file_paths)
        self.files_loaded = 0

        self.batch_size = batch_size

        self.desired_n_to_p_ratio = desired_n_to_p_ratio
        self.downsample = downsample

        self.cache = None
        self.cache_length = None
        self.cache_index = None

        self.parse_batches = True

    @staticmethod
    def get_all_dataset_paths(dataset_log_path):
        """
        Take a log file stating every npz file path and the number of rows (0-axis)

        file_path   length
        /path/to/file.npz   137

        and return paths, and their cumulative length

        :param dataset_log_path:
        :param train_size_proportion:
        :return: paths, length
        """
        dataset_log = pandas.read_csv(dataset_log_path, sep='\t')

        paths = list(dataset_log["file_path"])
        length = dataset_log["length"].sum()

        return paths, length

    @staticmethod
    def partition_dataset_paths(dataset_log_path, train_size_proportion):
        """
        Take a log file stating every npz file path and the number of rows (0-axis)

        file_path   length
        /path/to/file.npz   137

        and return 2 sets of paths, and their cumulative lengths. The partitioning of paths is  determined by the
        train_size_proportion parameter

        :param dataset_log_path:
        :param train_size_proportion:
        :return:
        """
        dataset_log = pandas.read_csv(dataset_log_path, sep='\t')
        total_length = dataset_log["length"].sum()

        # print("TOTAL LENGTH:", total_length)

        l = 0
        partition_index = None
        for i,length in enumerate(dataset_log["length"]):
            l += length

            if l >= round(float(total_length)*train_size_proportion):
                partition_index = i + 1
                break

        train_paths = list(dataset_log["file_path"][:partition_index])
        test_paths = list(dataset_log["file_path"][partition_index:])

        train_length = dataset_log["length"][:partition_index].sum()
        test_length = dataset_log["length"][partition_index:].sum()

        return train_paths, test_paths, train_length, test_length

    @staticmethod
    def get_region_from_file_path(file_path):
        basename = os.path.basename(file_path)
        basename = basename.split(".npz")[0]
        tokens = basename.split('_')

        chromosome, start, stop = tokens[-3:]
        start = int(start)
        stop = int(stop)

        return chromosome, start, stop

    @staticmethod
    def partition_dataset_paths_by_chromosome(dataset_log_path, test_chromosome_name_list):
        test_chromosome_name_set = set(test_chromosome_name_list)
        dataset_log = pandas.read_csv(dataset_log_path, sep='\t')
        train_paths = list()
        test_paths = list()
        train_length = 0
        test_length = 0

        for i,data in enumerate(zip(dataset_log["file_path"], dataset_log["length"])):
            path, length = data
            chromosome, start, stop = SplitDataloader.get_region_from_file_path(path)

            if chromosome in test_chromosome_name_set:
                test_paths.append(path)
                test_length += length
            else:
                train_paths.append(path)
                train_length += length

        return train_paths, test_paths, train_length, test_length

    def load_next_file(self):
        """
        Assuming there is another file in the list of paths, load it and concatenate with the leftover entries from last
        :return:
        """

        next_path = next(self.path_iterator)

        data = numpy.load(next_path)['a']
        data = data.T

        # remove any possible 1-size 3rd dimension
        if data.ndim > 2:
            data = data.squeeze()

        if self.downsample:
            data = self.downsample_negatives(data)

        if self.cache is not None:
            self.cache = numpy.concatenate([self.cache[self.cache_index:,:], data], axis=0)
        else:
            self.cache = data

        self.cache_length = self.cache.shape[0]
        self.cache_index = 0

        self.files_loaded += 1

    @staticmethod
    def parse_batch(batch):
        import torch

        x = batch[:,4:-1]
        y = batch[:,-1:]
        metadata = batch[:,:4]

        x_dtype = torch.FloatTensor
        y_dtype = torch.FloatTensor     # for MSE Loss or BCE loss
        # y_dtype = torch.LongTensor      # for CE Loss

        x = torch.from_numpy(x).type(x_dtype)
        y = torch.from_numpy(y).type(y_dtype)

        return x, y, metadata

    def downsample_negatives(self, cache):
        """
        In a table of data with terminal column of binary labels, subset the 0 rows based on desired 0:1 ratio
        :param cache:
        :return:
        """
        positive_mask = (cache[:,-1] == 1)
        negative_mask = numpy.invert(positive_mask)

        # find total number of positives and negatives
        n_positive = numpy.count_nonzero(positive_mask)
        n_negative = len(positive_mask) - n_positive

        # calculate downsampling coefficient for negative class (0)
        class_ratio = float(n_negative)/(n_positive+1e-5)
        c = min(1,self.desired_n_to_p_ratio/class_ratio)

        # generate a binomial vector with proportion of 1s equal to downsampling coefficient 'c'
        binomial_mask = numpy.random.binomial(1, c, len(positive_mask))

        # find intersection of binomial_vector and negative_mask
        negative_downsampling_mask = numpy.logical_and(negative_mask, binomial_mask)

        # find union of negative downsampling mask and the positive mask
        downsampling_mask = numpy.logical_or(negative_downsampling_mask,positive_mask)

        downsampled_cache = cache[downsampling_mask]

        return downsampled_cache

    def __next__(self):
        """
        Get the next batch data. DOES NOT RETURN FINAL BATCH IF != BATCH SIZE
        :return:
        """
        while self.cache_index + self.batch_size > self.cache_length:
            if self.files_loaded < self.n_files:
                self.load_next_file()
            else:
                raise StopIteration

        start = self.cache_index
        stop = self.cache_index + self.batch_size
        batch = self.cache[start:stop, :]

        self.cache_index += self.batch_size

        # assert(batch.shape[0] == self.batch_size)

        if self.parse_batches:
            batch = self.parse_batch(batch)

        return batch

    def __iter__(self):
        self.load_next_file()
        return self

