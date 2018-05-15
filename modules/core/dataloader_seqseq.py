import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
import pickle


class SequenceDataset(Dataset):
    """
    Arguments:
        A CSV file path
    """

    def __init__(self, csv_path, transform=None):
        data_frame = pd.read_csv(csv_path, header=None)
        assert data_frame[0].apply(lambda x: os.path.isfile(x)).all(), \
            "Some images referenced in the CSV file were not found"
        self.transform = transform

        self.regional_images = data_frame[0]
        self.allele_dictionary = data_frame[1]
        self.image_shape_y = data_frame[2]
        self.image_shape_x = data_frame[3]
        self.image_shape_z = data_frame[4]
        self.chromosome_name = data_frame[5]
        self.genomic_position_start = data_frame[6]
        self.image_slice_start = data_frame[7]
        self.image_slice_end = data_frame[8]
        self.label = data_frame[9]
        self.position_progress = data_frame[10]
        self.reference_sequences = data_frame[11]

    @staticmethod
    def load_dictionary(dictionary_location):
        f = open(dictionary_location, 'rb')
        dict = pickle.load(f)
        f.close()
        return dict

    def __getitem__(self, index):
        # load the image
        img = Image.open(self.regional_images[index])
        np_array_of_img = np.array(img.getdata())
        img.close()
        img_shape = (self.image_shape_y[index], self.image_shape_x[index], self.image_shape_z[index])
        img = np.reshape(np_array_of_img, img_shape)
        img = img[:, self.image_slice_start[index]:self.image_slice_end[index], :]
        img = np.transpose(img, (1, 0, 2))

        # load the labels
        label = [ord(x)-ord('A') for x in self.label[index][20:40]]
        label = np.array(label)

        # load genomic position information
        position_progress = self.position_progress[index][20:40]
        reference_sequence = self.reference_sequences[index][20:40]

        # load allele dictionary
        allele_dictionary = self.load_dictionary(self.allele_dictionary[index])

        # type fix and convert to tensor
        img = img.astype(dtype=np.uint8)
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(label)
        chromosome_name = self.chromosome_name[index]
        positional_information = (chromosome_name, position_progress, self.genomic_position_start[index],
                                  reference_sequence)

        return img, label, position_progress, positional_information, allele_dictionary

    def __len__(self):
        return len(self.regional_images.index)
