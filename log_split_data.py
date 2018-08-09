from modules.handlers.FileManager import FileManager
from modules.handlers.TsvWriter import TsvWriter
import pandas
import gzip
from tqdm import tqdm


parent_directory_path = "/home/ryan/data/Nanopore/filter_model_training_data/nanopore_chr1-8_2-2-5_perc_abs_cov_filtered/"
file_extension = ".pkl.gz"

file_paths = FileManager.get_all_file_paths_by_type(parent_directory_path=parent_directory_path,
                                                    file_extension=file_extension)

log_header = ["file_path", "length"]

log_writer = TsvWriter(output_directory=parent_directory_path,
                       header=log_header,
                       filename_prefix="dataset_log.tsv")

for path in tqdm(file_paths):
    file = gzip.open(path)
    data = pandas.read_pickle(file)
    length = data.shape[0]

    if data.size > 0:
        log_writer.append_row([path,length])

