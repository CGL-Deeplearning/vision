import argparse
import math
import time
import os
import sys
import multiprocessing
import h5py
from tqdm import tqdm
import numpy as np

from modules.core.FilterCandidateFinder import CandidateFinder
from modules.handlers.BamHandler import BamHandler
from modules.handlers.FastaHandler import FastaHandler
from modules.handlers.TextColor import TextColor
from modules.core.IntervalTree import IntervalTree
from modules.handlers.TsvHandler import TsvHandler
from modules.core.ImageGenerator import ImageGenerator
from modules.core.FilterCandidateVectorizer import CandidateVectorizer
from modules.handlers.FileManager import FileManager
from modules.handlers.TsvWriter import TsvWriter
from modules.core.filter_model import *

"""
This script creates prediction images from BAM and Reference FASTA. The process is:
- Find candidates that can be variants
- Create images for each candidate

Input:
- BAM file: Alignment of a genome
- REF file: The reference FASTA file used in the alignment
- BED file: A confident bed file. If confident_bed is passed it will only generate train set for those region.

Output:
- H5PY files: Containing images of the genome.
- CSV file: Containing records of images and their location in the H5PY file.
"""
DEBUG_PRINT_CANDIDATES = False
DEBUG_TIME_PROFILE = False
DEBUG_TEST_PARALLEL = False


def build_chromosomal_interval_trees(confident_bed_path):
    """
    Produce a dictionary of intervals trees, with one tree per chromosome
    :param confident_bed_path: Path to confident bed file
    :return: trees_chromosomal
    """
    tsv_handler_reference = TsvHandler(tsv_file_path=confident_bed_path)

    # universal_offset affects all coordinates, start_offset only affects start coordinates. BED format has open start
    # and closed stop for some reason e.g. (start,stop], so this needs to be converted to [start,stop]
    intervals_chromosomal_reference = tsv_handler_reference.get_bed_intervals_by_chromosome(start_offset=1,
                                                                                            universal_offset=-1)
    trees_chromosomal = dict()

    for chromosome_name in intervals_chromosomal_reference:
        intervals = intervals_chromosomal_reference[chromosome_name]
        tree = IntervalTree(intervals)

        trees_chromosomal[chromosome_name] = tree
    return trees_chromosomal


class View:
    """
    Process manager that runs sequence of processes to generate images and their labebls.
    """
    def __init__(self, chromosome_name, bam_file_path, reference_file_path, output_file_path, confident_tree):
        """
        Initialize a manager object
        :param chromosome_name: Name of the chromosome
        :param bam_file_path: Path to the BAM file
        :param reference_file_path: Path to the reference FASTA file
        :param output_file_path: Path to the output directory where images are saved
        :param confident_tree: Dictionary containing all confident trees. NULL if parameter not passed.
        """
        # --- initialize handlers ---
        self.bam_handler = BamHandler(bam_file_path)
        self.fasta_handler = FastaHandler(reference_file_path)
        self.output_dir = output_file_path
        self.confident_tree = confident_tree[chromosome_name] if confident_tree else None

        # --- initialize parameters ---
        self.chromosome_name = chromosome_name

    @staticmethod
    def get_images_for_two_alts(record):
        """
        Returns records for sites where we have two alternate alleles.
        :param record: Record that belong to the site
        :return: Records of a site
        """
        chr_name, pos_start, pos_end, ref, alt1, alt2, rec_type_alt1, rec_type_alt2 = record[0:8]
        # record for first allele
        rec_1 = [chr_name, pos_start, pos_end, ref, alt1, '.', rec_type_alt1, 0]
        # record for second allele
        rec_2 = [chr_name, pos_start, pos_end, ref, alt2, '.', rec_type_alt2, 0]
        # record for the image where both alleles are present
        rec_3 = [chr_name, pos_start, pos_end, ref, alt1, alt2, rec_type_alt1, rec_type_alt2]
        return [rec_1, rec_2, rec_3]

    def write_test_set(self, data, start, stop):
        """
        Create a npz training set of all labeled candidate sites found in the region
        :param data: vectors of the form [f1, f2, f3, ..., fn, L] where f = freq and L = label 0/1
        :param start: start coord of region
        :param stop: stop coord of region
        :return:
        """
        filename = "allele_frequencies_" + self.chromosome_name + "_" + str(start) + "_" + str(stop) + ".npz"
        path = (os.path.join(self.output_dir, filename))

        length = data.shape[0]
        if data.size > 0:
            np.savez_compressed(path, a=data)
            # self.log_writer.append_row([path, length])

    def in_confident_check(self, start, stop):
        """
        Check if an interval is inside the confident bed region.
        :param start: start position
        :param stop: stop position
        :return: Boolean, T: in confident, F: not in confident
        """
        interval = [int(start), int(stop)]

        # if interval is a subset, add it to output
        if self.confident_tree.contains_interval_subset(interval):
            return True
        return False

    def get_vectorized_candidate_sites(self, selected_candidate_list):
        """
        Label selected candidates of a region and return a list of records
        :param selected_candidate_list: List of all selected candidates with their alleles
        :param start_pos: start position of the region
        :param end_pos: end position of the region
        :param filter_hom_ref: whether to ignore hom_ref VCF records during candidate validation
        :return: labeled_sites: Labeled candidate sites. Each containing proper genotype.
        """
        # create an object for labeling the allele
        allele_labeler = CandidateVectorizer(fasta_handler=self.fasta_handler)

        vectorized_sites = allele_labeler.get_vectorized_candidates(chromosome_name=self.chromosome_name,
                                                                    candidate_sites=selected_candidate_list)

        return vectorized_sites

    def generate_candidate_images(self, candidate_list, image_generator, thread_no):
        """
        Generate images from a given labeled candidate list.
        :param candidate_list: List of candidates.
        :param image_generator: Image generator object containing all dictionaries to generate the images.
        :param thread_no: The thread number used to name the files.
        :return:
        """

        # declare the size of the image
        image_height, image_width = 300, 300
        if len(candidate_list) == 0:
            return

        # create summary file where the location of each image is recorded
        contig = str(self.chromosome_name)
        smry = open(self.output_dir + "summary/" + "summary" + '_' + contig + "_" + str(thread_no) + ".csv", 'w')

        # create a h5py file where the images are stored
        hdf5_filename = self.output_dir + contig + '_' + str(thread_no) + ".h5"
        hdf5_file = h5py.File(hdf5_filename, mode='w')

        # list of image records to be generated
        image_record_set = []

        # expand the records for sites where two alleles are found
        for record in candidate_list:
            chr_name, pos_start, pos_end, ref, alt1, alt2, rec_type_alt1, rec_type_alt2 = record[0:8]

            if alt2 != '.':
                image_record_set.extend(self.get_images_for_two_alts(record))
            else:
                image_record_set.append([chr_name, pos_start, pos_end, ref, alt1, alt2, rec_type_alt1, rec_type_alt2])

        # set of images we are generating
        img_set = []

        # index of the image we generate the images
        indx = 0
        for img_record in image_record_set:
            chr_name, pos_start, pos_end, ref, alt1, alt2, rec_type_alt1, rec_type_alt2 = img_record[0:8]

            # list of alts in this record
            alts = [alt1]

            # list of type of record (IN, DEL, SNP)
            rec_types = [rec_type_alt1]
            if alt2 != '.':
                alts.append(alt2)
                rec_types.append(rec_type_alt2)

            # the image array
            image_array = image_generator.create_image(pos_start, ref, alts, rec_types,
                                                       image_height=image_height, image_width=image_width)

            # the record of the image we want to save in the summary file
            img_rec = str('\t'.join(str(item) for item in img_record))
            img_set.append(np.array(image_array, dtype=np.int8))
            smry.write(os.path.abspath(hdf5_filename) + ',' + str(indx) + ',' + img_rec + '\n')
            indx += 1

        # the image dataset we save. The index name in h5py is "images".
        img_dset = hdf5_file.create_dataset("images", (len(img_set),) + (image_height, image_width, 7), np.int8, compression='gzip')

        # save the images
        img_dset[...] = img_set

    def subset_by_confident_region(self, candidates):
        confident_candidates = []

        for candidate in candidates:
            pos_st = candidate[1] + 1   # offset for bed
            pos_end = candidate[1] + 1  # offset for bed
            in_conf = self.in_confident_check(pos_st, pos_end)
            if in_conf is True:
                confident_candidates.append(candidate)

        return confident_candidates

    @staticmethod
    def build_interval_tree_from_coordinate_data(coordinates):
        intervals = list()
        for i in range(coordinates.shape[0]):
            # chromosome_number = coordinates[i,0]
            start_position = int(coordinates[i,1])

            interval = [start_position, start_position]
            intervals.append(interval)

        interval_tree = IntervalTree(intervals)

        return interval_tree

    def predict_true_candidates(self, vectorized_candidates, candidates):
        # specify path to model state file
        model_state_file_path = "models/filter_model_state"

        # predict
        positive_coordinates = predict_sites(model_state_file_path, vectorized_candidates)

        positive_coordinates_list = list(map(int,list(positive_coordinates[:,1])))
        positive_coordinates_set = set(positive_coordinates_list)

        # generate interval tree from table of positive
        interval_tree = self.build_interval_tree_from_coordinate_data(positive_coordinates)

        # use interval tree to subset coordinates
        predicted_candidates = list()
        for candidate in candidates:
            start = candidate[1]

            interval = [start, start]

            if interval in interval_tree:
                predicted_candidates.append(candidate)

            # if start in positive_coordinates_set:
            #     predicted_candidates.append(candidate)

        # for candidate in predicted_candidates:
        #     print(candidate)

        return predicted_candidates

    def parse_region(self, start_position, end_position, thread_no):
        """
        Generate labeled images of a given region of the genome
        :param start_position: Start position of the region
        :param end_position: End position of the region
        :param thread_no: Thread no for this region
        :return:
        """
        # get the reads that fall in that region
        reads = self.bam_handler.get_reads(chromosome_name=self.chromosome_name,
                                           start=start_position,
                                           stop=end_position)

        # create candidate finder object
        candidate_finder = CandidateFinder(reads=reads,
                                           fasta_handler=self.fasta_handler,
                                           chromosome_name=self.chromosome_name,
                                           region_start_position=start_position,
                                           region_end_position=end_position)

        # go through each read and find candidate positions and alleles
        candidates = candidate_finder.parse_reads_and_select_candidates(reads=reads)
        # dictionaries_for_images = candidate_finder.get_pileup_dictionaries()

        # if confident tree is defined then subset the candidates to only those intervals
        if self.confident_tree is not None:
            candidates = self.subset_by_confident_region(candidates)

        # get all vectorized candidate sites
        vectorized_candidates = self.get_vectorized_candidate_sites(candidates)

        if vectorized_candidates.size > 0:
            # pass test set to trained model and get predicted candidates
            predicted_candidates = self.predict_true_candidates(vectorized_candidates, candidates)

            # print("candidates:", len(candidates))
            # print("predicted:", len(predicted_candidates))

            if DEBUG_PRINT_CANDIDATES:
                for candidate in predicted_candidates:
                    print(candidate)

            # # create image generator object with all necessary dictionary
            # image_generator = ImageGenerator(dictionaries_for_images)
            #
            # # generate and save candidate images
            # self.generate_candidate_images(predicted_candidates, image_generator, thread_no)


def parallel_run(chr_name, bam_file, ref_file, output_dir, start_pos, end_pos, conf_bed_tree, thread_no):
    """
    Creates a view object for a region and generates images for that region.
    :param chr_name: Name of the chromosome
    :param bam_file: path to BAM file
    :param ref_file: path to reference FASTA file
    :param output_dir: path to output directory
    :param start_pos: start position of the genomic region
    :param end_pos: end position of the genomic region
    :param conf_bed_tree: tree containing confident bed intervals
    :param thread_no: thread number
    :return:
    """

    # create a view object
    view_ob = View(chromosome_name=chr_name,
                   bam_file_path=bam_file,
                   reference_file_path=ref_file,
                   output_file_path=output_dir,
                   confident_tree=conf_bed_tree)

    # return the results
    view_ob.parse_region(start_pos, end_pos, thread_no)


def create_output_dir_for_chromosome(output_dir, chr_name):
    """
    Create an internal directory inside the output directory to dump choromosomal summary files
    :param output_dir: Path to output directory
    :param chr_name: chromosome name
    :return: New directory path
    """
    path_to_dir = output_dir + chr_name + "/"
    if not os.path.exists(path_to_dir):
        os.mkdir(path_to_dir)

    summary_path = path_to_dir + "summary" + "/"
    if not os.path.exists(summary_path):
        os.mkdir(summary_path)

    return path_to_dir


def log_candidate_datasets(parent_directory_path):
    file_extension = ".npz"
    file_paths = FileManager.get_all_filepaths_by_type(parent_directory_path=parent_directory_path,
                                                       file_extension=file_extension)

    log_header = ["file_path", "length"]
    log_writer = TsvWriter(output_directory=parent_directory_path,
                           header=log_header,
                           filename_prefix="candidate_dataset_log.tsv")

    for path in tqdm(file_paths):
        data = np.load(path)['a']
        length = data.shape[1]

        if data.size > 0:
            log_writer.append_row([path, length])


def chromosome_level_parallelization(chr_name, bam_file, ref_file, output_path, max_threads, confident_bed_tree):
    """
    This method takes one chromosome name as parameter and chunks that chromosome in max_threads.
    :param chr_name: Name of the chromosome
    :param bam_file: path to BAM file
    :param ref_file: path to reference FASTA file
    :param output_path: path to output directory
    :param max_threads: Maximum number of threads to run at one instance
    :param confident_bed_tree: tree containing confident bed intervals
    :return:
    """
    # create dump directory inside output directory
    output_dir = create_output_dir_for_chromosome(output_path, chr_name)

    # entire length of chromosome
    fasta_handler = FastaHandler(ref_file)
    whole_length = fasta_handler.get_chr_sequence_length(chr_name)

    # 2MB segments at once
    each_segment_length = 50000

    # chunk the chromosome into 1000 pieces
    chunks = int(math.ceil(whole_length / each_segment_length))
    if DEBUG_TEST_PARALLEL:
        chunks = 4
    for i in tqdm(range(chunks)):
        start_position = i * each_segment_length
        end_position = min((i + 1) * each_segment_length, whole_length)
        args = (chr_name, bam_file, ref_file, output_dir, start_position, end_position, confident_bed_tree, i)

        p = multiprocessing.Process(target=parallel_run, args=args)
        p.start()

        while True:
            if len(multiprocessing.active_children()) < max_threads:
                break


def genome_level_parallelization(bam_file, ref_file, output_dir_path, max_threads, confident_bed_tree):
    """
    This method calls chromosome_level_parallelization for each chromosome.
    :param bam_file: path to BAM file
    :param ref_file: path to reference FASTA file
    :param output_dir_path: path to output directory
    :param max_threads: Maximum number of threads to run at one instance
    :param confident_bed_tree: tree containing confident bed intervals
    :return:
    """
    # chr_list = ["chr1", "chr2", "chr3", "chr4", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11",
    #             "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22"]
    # chr_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
    # "19"]
    # --- NEED WORK HERE --- GET THE CHROMOSOME NAMES FROM THE BAM FILE
    chr_list = ["19"]
    program_start_time = time.time()

    # each chromosome in list
    for chr_name in chr_list:
        sys.stderr.write(TextColor.BLUE + "STARTING " + str(chr_name) + " PROCESSES" + "\n")
        start_time = time.time()

        # do a chromosome level parallelization
        chromosome_level_parallelization(chr_name, bam_file, ref_file, output_dir_path, max_threads, confident_bed_tree)

        end_time = time.time()
        sys.stderr.write(TextColor.PURPLE + "FINISHED " + str(chr_name) + " PROCESSES" + "\n")
        sys.stderr.write(TextColor.CYAN + "TIME ELAPSED: " + str(end_time - start_time) + "\n")

    # wait for the last process to end before file processing
    while True:
        if len(multiprocessing.active_children()) == 0:
            break

    for chr_name in chr_list:
        # here we dumped all the bed files
        path_to_dir = output_dir_path + chr_name + "/summary/"

        concatenated_file_name = output_dir_path + chr_name + ".csv"

        filemanager_object = FileManager()
        # get all bed file paths from the directory
        file_paths = filemanager_object.get_file_paths_from_directory(path_to_dir)
        # dump all bed files into one
        filemanager_object.concatenate_files(file_paths, concatenated_file_name)
        # delete all temporary files
        filemanager_object.delete_files(file_paths)
        # remove the directory
        os.rmdir(path_to_dir)

    program_end_time = time.time()
    sys.stderr.write(TextColor.RED + "PROCESSED FINISHED SUCCESSFULLY" + "\n")
    sys.stderr.write(TextColor.CYAN + "TOTAL TIME FOR GENERATING ALL RESULTS: " + str(program_end_time-program_start_time) + "\n")


def test(view_object):
    """
    Run a test
    :return:
    """
    start_time = time.time()
    # view_object.parse_region(start_position=1521297, end_position=1521302, thread_no=1)
    view_object.parse_region(start_position=3039222, end_position=3039224, thread_no=1)
    print("TOTAL TIME ELAPSED: ", time.time()-start_time)


def handle_output_directory(output_dir):
    """
    Process the output directory and return a valid directory where we save the output
    :param output_dir: Output directory path
    :return:
    """
    # process the output directory
    if output_dir[-1] != "/":
        output_dir += "/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # create an internal directory so we don't overwrite previous runs
    timestr = time.strftime("%m%d%Y_%H%M%S")
    internal_directory = "run_" + timestr + "/"
    output_dir = output_dir + internal_directory

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    return output_dir


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks.
    '''
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--bam",
        type=str,
        required=True,
        help="BAM file containing reads of interest."
    )
    parser.add_argument(
        "--ref",
        type=str,
        required=True,
        help="Reference corresponding to the BAM file."
    )
    parser.add_argument(
        "--chromosome_name",
        type=str,
        help="Desired chromosome number E.g.: 3"
    )
    parser.add_argument(
        "--max_threads",
        type=int,
        default=5,
        help="Number of maximum threads for this region."
    )
    parser.add_argument(
        "--confident_bed",
        type=str,
        default='',
        help="Path to confident BED file"
    )
    parser.add_argument(
        "--test",
        type=bool,
        default=False,
        help="If true then a dry test is run."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="candidate_finder_output/",
        help="Path to output directory."
    )

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.output_dir = handle_output_directory(FLAGS.output_dir)

    # if the confident bed is not empty then create the tree
    if FLAGS.confident_bed != '':
        confident_tree_build = build_chromosomal_interval_trees(FLAGS.confident_bed)
    else:
        confident_tree_build = None

    if confident_tree_build is not None:
        sys.stderr.write(TextColor.PURPLE + "CONFIDENT TREE LOADED\n" + TextColor.END)
    else:
        sys.stderr.write(TextColor.RED + "CONFIDENT BED IS NULL\n" + TextColor.END)

    if FLAGS.test is True:
        chromosome_output = create_output_dir_for_chromosome(FLAGS.output_dir, FLAGS.chromosome_name)
        view = View(chromosome_name=FLAGS.chromosome_name,
                    bam_file_path=FLAGS.bam,
                    reference_file_path=FLAGS.ref,
                    output_file_path=chromosome_output,
                    confident_tree=confident_tree_build)
        test(view)
    elif FLAGS.chromosome_name is not None:
        chromosome_level_parallelization(FLAGS.chromosome_name, FLAGS.bam, FLAGS.ref, FLAGS.output_dir,
                                         FLAGS.max_threads, confident_tree_build)
    else:
        genome_level_parallelization(FLAGS.bam, FLAGS.ref, FLAGS.output_dir,
                                     FLAGS.max_threads, confident_tree_build)
