import os
import numpy as np
import h5py
import pandas as pd
from torch.utils.data import Dataset
import sys


class TextColor:
    """
    Defines color codes for text used to give different mode of errors.
    """
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class PileupDataset(Dataset):
    """
    Creates a pile up image from hdf5 file as specified by csv file.
    Arguments:
        A CSV file path
    """

    def __init__(self, csv_path, transform=None):
        tmp_df = pd.read_csv(csv_path, header=None)
        for img_file in tmp_df[0]:
            if os.path.isfile(img_file) is False:
                print("Some images referenced in the CSV file were not found: ", img_file)
                exit()
        # assert tmp_df[0].apply(lambda x: os.path.isfile(x)).all(), \
        #     "Some images referenced in the CSV file were not found: "
        self.transform = transform

        self.X_train = tmp_df[0]
        self.X_train_index = tmp_df[1]
        self.rec = tmp_df[2]

    def __getitem__(self, index):
        hdf5_file_path = self.X_train[index]
        indx = self.X_train_index[index]
        rec = self.rec[index]
        hdf5_file = h5py.File(hdf5_file_path, 'r')

        if 'images' not in hdf5_file.keys():
            print("NO IMAGES IN HDF5", hdf5_file_path, rec)
        if 'labels' not in hdf5_file.keys():
            print("NO LABELS IN HDF5", hdf5_file_path, rec)

        image_dataset = hdf5_file['images']
        label_dataset = hdf5_file['labels']

        img = image_dataset[indx]
        label = int(label_dataset[indx])

        img = img.astype(dtype=np.uint8)
        if self.transform is not None:
            img = self.transform(img)
            img = img.transpose(1, 2)

        return img, label, rec

    def __len__(self):
        return len(self.X_train.index)