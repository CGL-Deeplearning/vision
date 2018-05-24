from modules.handlers.FileManager import FileManager
import os
import numpy
import pandas
import random
import gzip

class SplitDataloader:
    def __init__(self, file_paths, length, batch_size, desired_n_to_p_ratio=20, downsample=False, use_gpu=False):
        self.file_paths = file_paths
        self.path_iterator = iter(file_paths)
        self.length = length
        self.n_files = len(file_paths)
        self.files_loaded = 0

        self.use_gpu = use_gpu

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
        basename = basename.split(".pkl.gz")[0]
        tokens = basename.split('_')

        chromosome, start, stop = tokens[-3:]
        start = int(start)
        stop = int(stop)

        return chromosome, start, stop

    @staticmethod
    def partition_dataset_paths_by_chromosome(dataset_log_path, test_chromosome_name_list, train_chromosome_list=None):
        test_chromosome_name_set = set(test_chromosome_name_list)
        if train_chromosome_list is not None:
            train_chromosome_set = set(train_chromosome_list)
        else:
            train_chromosome_set = None

        dataset_log = pandas.read_csv(dataset_log_path, sep='\t')
        train_paths = list()
        test_paths = list()
        train_length = 0
        test_length = 0

        for i,data in enumerate(zip(dataset_log["file_path"], dataset_log["length"])):
            path, length = data
            chromosome, start, stop = SplitDataloader.get_region_from_file_path(path)

            # append the TEST list
            if chromosome in test_chromosome_name_set:
                test_paths.append(path)
                test_length += length

            # append the TRAINING list specifically if contained in training chromosome set (if specified)
            elif train_chromosome_set is not None:
                if chromosome in train_chromosome_set:
                    train_paths.append(path)
                    train_length += length

            # append the TRAINING list if no training chromosome list is specified
            else:
                train_paths.append(path)
                train_length += length

        return train_paths, test_paths, train_length, test_length

    def load_next_numpy_file(self):
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

    def load_next_pandas_file(self):
        """
        Assuming there is another file in the list of paths, load it and concatenate with the leftover entries from last
        :return:
        """
        next_path = next(self.path_iterator)

        file = gzip.open(next_path)
        data = pandas.read_pickle(file)
        # length = data.shape[0]
        # data = data.T

        if self.downsample:
            print("WARNING: Downsampling not implemented for Pandas datasets")

        if self.cache is not None:
            self.cache = pandas.concat([self.cache[self.cache_index:], data], axis=0)
        else:
            self.cache = data

        self.cache_length = self.cache.shape[0]
        self.cache_index = 0

        self.files_loaded += 1

    @staticmethod
    def parse_numpy_batch(batch):
        import torch
        from torch.autograd import Variable

        x = batch[:,4:-1]
        y = batch[:,-1:]
        metadata = batch[:,:4]

        x_dtype = torch.FloatTensor
        y_dtype = torch.FloatTensor     # for MSE Loss or BCE loss
        # y_dtype = torch.LongTensor      # for CE Loss

        x = torch.from_numpy(x).type(x_dtype)
        y = torch.from_numpy(y).type(y_dtype)

        return x, y, metadata

    def parse_pandas_batch(self, batch):
        import torch
        from torch.autograd import Variable

        metadata_headers = ["chromosome_number", "position", "genotype_1", "genotype_2"]
        x_data_headers = ["frequency_1", "frequency_2", "frequency_3", "frequency_4", "frequency_5", "frequency_6",
                          "frequency_7", "frequency_8", "frequency_9", "frequency_10", "frequency_11", "frequency_12",
                          "coverage", "map_quality_ref_1", "map_quality_ref_2", "map_quality_ref_3",
                          "map_quality_ref_4", "map_quality_ref_5", "map_quality_ref_6", "base_quality_ref_1",
                          "base_quality_ref_2", "base_quality_ref_3", "base_quality_ref_4", "base_quality_ref_5",
                          "base_quality_ref_6", "map_quality_non_ref_1", "map_quality_non_ref_2",
                          "map_quality_non_ref_3", "map_quality_non_ref_4", "map_quality_non_ref_5",
                          "map_quality_non_ref_6", "base_quality_non_ref_1", "base_quality_non_ref_2",
                          "base_quality_non_ref_3", "base_quality_non_ref_4", "base_quality_non_ref_5",
                          "base_quality_non_ref_6"]
        y_data_headers = ["label"]

        x = batch[x_data_headers]
        y = batch[y_data_headers]
        metadata = batch[metadata_headers]

        # print("x")
        # print(x)
        # print("y")
        # print(y)
        # print("metadata")
        # print(metadata)
        # exit()

        if self.use_gpu:
            x_dtype = torch.cuda.FloatTensor
            y_dtype = torch.cuda.FloatTensor  # for MSE Loss or BCE loss
        else:
            x_dtype = torch.FloatTensor
            y_dtype = torch.FloatTensor  # for MSE Loss or BCE loss
            # y_dtype = torch.LongTensor    # for CE Loss

        x = x_dtype(x.values)
        y = y_dtype(y.values)

        if self.use_gpu:
            x.cuda()
            y.cuda()

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
        c = min(1, self.desired_n_to_p_ratio/class_ratio)

        # generate a binomial vector with proportion of 1s equal to downsampling coefficient 'c'
        binomial_mask = numpy.random.binomial(1, c, len(positive_mask))

        # find intersection of binomial_vector and negative_mask
        negative_downsampling_mask = numpy.logical_and(negative_mask, binomial_mask)

        # find union of negative downsampling mask and the positive mask
        downsampling_mask = numpy.logical_or(negative_downsampling_mask, positive_mask)

        downsampled_cache = cache[downsampling_mask]

        return downsampled_cache

    def __next__(self):
        """
        Get the next batch data. DOES NOT RETURN FINAL BATCH IF != BATCH SIZE
        :return:
        """
        while self.cache_index + self.batch_size > self.cache_length:
            if self.files_loaded < self.n_files:
                self.load_next_pandas_file()
            else:
                raise StopIteration

        start = self.cache_index
        stop = self.cache_index + self.batch_size

        # print(start,stop)
        batch = self.cache[start:stop]

        self.cache_index += self.batch_size

        assert(batch.shape[0] == self.batch_size)

        if self.parse_batches:
            batch = self.parse_pandas_batch(batch)

        return batch

    def __iter__(self):
        self.load_next_pandas_file()
        return self

