import os
from PIL import Image, ImageOps
import numpy as np
import torch
import h5py
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms, utils
from sklearn.preprocessing import MultiLabelBinarizer
from torch.autograd import Variable
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
    Arguments:
        A CSV file path
    """

    def __init__(self, csv_path, transform=None):
        tmp_df = pd.read_csv(csv_path, header=None)
        assert tmp_df[0].apply(lambda x: os.path.isfile(x)).all(), \
            "Some images referenced in the CSV file were not found"

        self.mlb = MultiLabelBinarizer()
        self.transform = transform

        self.X_train = tmp_df[0]
        self.X_train_index = tmp_df[1]
        # labelLists = []
        # for label in tmp_df[2]:
        #     labelList = [int(x) for x in str(label)]
        #     labelLists.append(np.array(labelList, dtype=np.long))
        self.y_train = tmp_df[2]
        self.rec = tmp_df[3]

    def __getitem__(self, index):
        hdf5_file_path = self.X_train[index]
        indx = self.X_train_index[index]
        label_csv = self.y_train[index]
        rec = self.rec[index]
        hdf5_file = h5py.File(hdf5_file_path, 'r')
        image_dataset = hdf5_file['images']
        label_dataset = hdf5_file['labels']

        img = image_dataset[indx]
        label = label_dataset[indx]
        hdf5_file.close()

        img = img.astype(dtype=np.uint8)
        if self.transform is not None:
            img = self.transform(img)

        if label != label_csv:
            sys.stderr.write("CSV LABEL DID NOT MATCH HDF5 LABEL" + str(rec) + "\n")

        return img, label_csv, rec

    def __len__(self):
        return len(self.X_train.index)
