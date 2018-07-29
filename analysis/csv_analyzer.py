import pandas as pd
import os
import sys
import h5py
import tqdm as tqdm

csv_path = sys.argv[1]

tmp_df = pd.read_csv(csv_path, header=None)

for i in tqdm(range(len(tmp_df[0]))):
    img_file = tmp_df[0][i]
    if os.path.isfile(img_file) is False:
        print("INVALID FILE PATH: ", img_file)
        exit()
    else:
        hdf5_file = h5py.File(img_file, 'r')

        if 'images' not in hdf5_file.keys():
            print("NO IMAGES IN HDF5", img_file)
        if 'labels' not in hdf5_file.keys():
            print("NO LABELS IN HDF5", img_file)
