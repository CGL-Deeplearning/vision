import argparse
import h5py
import sys
sys.path.append(".")
import modules.core.ImageAnalyzer as image_analyzer


def extract_bed(hdf5_file_path):
    hdf5_file = h5py.File(hdf5_file_path, 'r')
    record_dataset = hdf5_file['records']
    for i, record in enumerate(record_dataset):
        print(i, record)

def analyze_image(hdf5_file_path, index):
    hdf5_file = h5py.File(hdf5_file_path, 'r')
    image_dataset = hdf5_file['images']
    record_dataset = hdf5_file['records']
    label_dataset = hdf5_file['labels']
    image = image_dataset[index]
    label = label_dataset[index]
    record = record_dataset[index]
    print(label, record)
    image_analyzer.analyze_np_array(image, image.shape[0], image.shape[1])


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks to generate the pileup.
    '''
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--hdf5_file",
        type=str,
        required=True,
        help="Bed file containing confident windows."
    )
    parser.add_argument(
        "--extract_bed",
        type=bool,
        required=False,
        default=False,
        help="Bed file containing confident windows."
    )
    parser.add_argument(
        "--analyze_img",
        type=bool,
        required=False,
        default=False,
        help="Bed file containing confident windows."
    )
    parser.add_argument(
        "--index",
        type=int,
        required=False,
        default=0,
        help="Bed file containing confident windows."
    )
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.extract_bed:
        extract_bed(FLAGS.hdf5_file)
    if FLAGS.analyze_img:
        analyze_image(FLAGS.hdf5_file, FLAGS.index)