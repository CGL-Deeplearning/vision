from modules.handlers.FileManager import FileManager
from modules.handlers.TsvWriter import TsvWriter
import pandas
import gzip
from tqdm import tqdm

parent_directory_path = "/home/ryan/data/Nanopore/filter_model_training_data/nanopore_chr1-22_2-2-5_perc-abs-cov_filtered/"
file_extension = ".pkl.gz"

file_paths = FileManager.get_all_file_paths_by_type(parent_directory_path=parent_directory_path,
                                                    file_extension=file_extension)

data_tables = list()

for p,path in enumerate(file_paths):
    file = gzip.open(path)
    data = pandas.read_pickle(file)

    if data.size > 0:
        data_tables.append(data)

    if p > 20:
        break

    print(p)

concatenated_data = pandas.concat(data_tables)

concatenated_data.to_csv("nanopore_labeled_filtered_sites.tsv", sep='\t')
