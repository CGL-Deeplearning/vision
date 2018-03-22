import torch
from modules.handlers.bed2train import get_prediction_set_from_bed, get_flattened_bed
from modules.core.TrainBed2Image import TrainBed2ImageAPI
from torch.utils.data import Dataset
import time


class DataSetLoader(Dataset):
    def __init__(self, bam_file_path, fasta_file_path, bed_file_path, transform):
        self.bam_file_path = bam_file_path
        self.fasta_file_path = fasta_file_path

        self.all_bed_records = get_prediction_set_from_bed(bed_file_path)
        self.all_bed_records = get_flattened_bed(self.all_bed_records)

        self.transform = transform

        self.generated_files = {}

    def __getitem__(self, index):
        bed_record = self.all_bed_records[index]

        api_object = TrainBed2ImageAPI(self.bam_file_path, self.fasta_file_path)
        img, img_shape, label = api_object.create_image(api_object.bam_handler, api_object.fasta_handler, bed_record)
        label = torch.LongTensor([int(label)])

        if self.transform is not None:
            img = self.transform(img)
        return img, label, bed_record

    def __len__(self):
        return len(self.all_bed_records)
