from modules.handlers.FileManager import FileManager
from modules.handlers.TsvWriter import TsvWriter
from modules.core.SplitDatasetLoader import SplitDataloader
import os
import numpy
import pandas
import gzip
import random


# --- TEST PANDAS DATALOADER ---

output_dir_path = "test"

for i in range(10):
    a = numpy.arange(i*10,(i+1)*10)
    a = a.reshape(a.shape[0],1)
    a = numpy.concatenate([a,a,a], axis=1)

    pandas_data_frame = pandas.DataFrame(a)

    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)

    output_file_path = os.path.join(output_dir_path, "test_"+str(i)+".pkl.gz")

    pandas_data_frame.to_pickle(output_file_path, compression="gzip")
    print(a)


file_extension = ".pkl.gz"

file_paths = FileManager.get_all_file_paths_by_type(parent_directory_path=output_dir_path,
                                                    file_extension=file_extension)

log_header = ["file_path", "length"]

log_writer = TsvWriter(output_directory=output_dir_path,
                       header=log_header,
                       filename_prefix="dataset_log.tsv")


for path in file_paths:
    file = gzip.open(path)
    data = pandas.read_pickle(file)
    length = data.shape[0]

    if data.size > 0:
        log_writer.append_row([path,length])
        print([path, length])


dataset_log_path = os.path.join(output_dir_path, "dataset_log.tsv")

train_paths, test_paths, train_length, test_length = SplitDataloader.partition_dataset_paths(dataset_log_path=dataset_log_path, train_size_proportion=0.4)


print(train_length)
print(train_paths)
print(test_length)
print(test_paths)

dataloader = SplitDataloader(file_paths=train_paths, length=train_length, batch_size=8)
dataloader.parse_batches = False

print("train")
for i, batch in enumerate(dataloader):
    print(batch.values)
    print(i)

dataloader = SplitDataloader(file_paths=test_paths, length=test_length, batch_size=8)
dataloader.parse_batches = False

print("test")
for i, batch in enumerate(dataloader):
    print(batch.values)
    print(i)


# --- TEST NUMPY DATALOADER ---

# output_dir_path = "test"
#
# for i in range(10):
#     a = numpy.arange(i*10,(i+1)*10)
#     a = a.reshape(a.shape[0],1)
#     a = numpy.concatenate([a,a,a], axis=1)
#
#     if not os.path.exists(output_dir_path):
#         os.mkdir(output_dir_path)
#
#     output_file_path = os.path.join(output_dir_path, "test_"+str(i))
#
#     numpy.savez_compressed(output_file_path, a=a)
#     print(a)
#
#
# file_extension = ".pkl.gz"
#
# file_paths = FileManager.get_all_file_paths_by_type(parent_directory_path=output_dir_path,
#                                                     file_extension=file_extension)
#
# log_header = ["file_path", "length"]
#
# log_writer = TsvWriter(output_directory=output_dir_path,
#                        header=log_header,
#                        filename_prefix="dataset_log.tsv")
#
#
# for path in file_paths:
#     data = numpy.load(path)['a']
#     length = data.shape[0]
#
#     if data.size > 0:
#         log_writer.append_row([path,length])
#         print([path,length])
#
#
# dataset_log_path = os.path.join(output_dir_path, "dataset_log.tsv")
#
# train_paths, test_paths, train_length, test_length = SplitDataloader.partition_dataset_paths(dataset_log_path=dataset_log_path, train_size_proportion=0.4)
#
#
# print(train_length)
# print(train_paths)
# print(test_length)
# print(test_paths)
#
# dataloader = SplitDataloader(file_paths=train_paths, length=train_length, batch_size=8)
# dataloader.parse_batches = False
#
# print("train")
# for i, batch in enumerate(dataloader):
#     print(batch)
#
# dataloader = SplitDataloader(file_paths=test_paths, length=test_length, batch_size=8)
# dataloader.parse_batches = False
#
# print("test")
# for i, batch in enumerate(dataloader):
#     print(batch)


# --- TEST DOWNSAMPLING ---

# a = numpy.array([[0,0,0,0,0,0,0,0,1,1],[0,0,0,0,0,0,0,0,1,1]]).T
# print(a)
# print(a.shape)
#
# total_positive = 0
# total_negative = 0
# for i in range(20):
#     b = SplitDataloader.downsample_negatives(a)
#
#     # ----------------------------------------------
#     positive_mask = (b[:, -1] == 1)
#     negative_mask = numpy.invert(positive_mask)
#
#     # find total number of positives and negatives
#     n_positive = numpy.count_nonzero(positive_mask)
#     n_negative = len(positive_mask) - n_positive
#
#
#     total_positive += n_positive
#     total_negative += n_negative
#
#     print(n_positive)
#     print(n_negative)
#     # ----------------------------------------------
#
# print(total_positive)
# print(total_negative)

