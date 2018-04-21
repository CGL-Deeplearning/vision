import argparse
import math
import time
import os
import sys
import multiprocessing
import h5py
from tqdm import tqdm
import numpy as np

from modules.core.CandidateFinder import CandidateFinder
from modules.handlers.BamHandler import BamHandler
from modules.handlers.FastaHandler import FastaHandler
from modules.handlers.TextColor import TextColor
from modules.core.IntervalTree import IntervalTree
from modules.handlers.TsvHandler import TsvHandler
from modules.core.ImageGenerator import ImageGenerator
from modules.handlers.FileManager import FileManager
"""
candidate_finder finds possible variant sites in given bam file.
This script selects candidates for variant calling. 
It walks through the genome, looks at mismatches and records candidates.

It requires three parameters:
- bam_file_path: path to a bam file
- reference_file_path: path to a reference file
- vcf_file_path: path to a VCF file for true genotype labeling

Creates:
- Bed files containing candidate sites.


Also, the terms "window" and "region" are NOT interchangeable.
Region: A genomic region of interest where we want to find possible variant candidate
Window: A window in genomic region where there can be multiple alleles

A region can have multiple windows and each window belongs to a region.
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
    # what's the point of the offset? they sum to 0
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
    Works as a main class and handles user interaction with different modules.
    """
    def __init__(self, chromosome_name, bam_file_path, reference_file_path, output_file_path, confident_tree):
        # --- initialize handlers ---
        self.bam_handler = BamHandler(bam_file_path)
        self.fasta_handler = FastaHandler(reference_file_path)
        self.output_dir = output_file_path
        self.confident_tree = confident_tree[chromosome_name] if confident_tree else None

        # --- initialize parameters ---
        self.chromosome_name = chromosome_name

    @staticmethod
    def get_images_for_two_alts(record):
        chr_name, pos_start, pos_end, ref, alt1, alt2, rec_type_alt1, rec_type_alt2 = record[0:8]

        rec_1 = [chr_name, pos_start, pos_end, ref, alt1, '.', rec_type_alt1, 0]
        rec_2 = [chr_name, pos_start, pos_end, ref, alt2, '.', rec_type_alt2, 0]
        rec_3 = [chr_name, pos_start, pos_end, ref, alt1, alt2, rec_type_alt1, rec_type_alt2]
        return [rec_1, rec_2, rec_3]

    def in_confident_check(self, start, stop):
        """
        Test a raw BED file against a dictionary of interval trees based on chromosome, output list of lists (BED format)
        :param start: start position
        :param stop: stop position
        :return: Boolean, T: in confident, F: not in confident
        """
        interval = [int(start), int(stop)]

        # if interval is a subset, add it to output
        if self.confident_tree.contains_interval_subset(interval):
            return True
        return False

    def generate_candidate_images(self, candidate_list, image_generator, thread_no):
        image_height, image_width = 300, 300
        if len(candidate_list) == 0:
            return
        contig = str(self.chromosome_name)
        smry = open(self.output_dir + "summary/" + "summary" + '_' + contig + "_" + str(thread_no) + ".csv", 'w')
        hdf5_filename = self.output_dir + contig + '_' + str(thread_no) + ".h5"
        hdf5_file = h5py.File(hdf5_filename, mode='w')
        image_set = []
        for record in candidate_list:
            chr_name, pos_start, pos_end, ref, alt1, alt2, rec_type_alt1, rec_type_alt2 = record[0:8]

            if alt2 != '.':
                image_set.extend(self.get_images_for_two_alts(record))
            else:
                image_set.append([chr_name, pos_start, pos_end, ref, alt1, alt2, rec_type_alt1, rec_type_alt2])

        img_set = []
        indx = 0
        for img_record in image_set:
            chr_name, pos_start, pos_end, ref, alt1, alt2, rec_type_alt1, rec_type_alt2 = img_record[0:8]

            alts = [alt1]
            rec_types = [rec_type_alt1]
            if alt2 != '.':
                alts.append(alt2)
                rec_types.append(rec_type_alt2)
            image_array = image_generator.create_image(pos_start, ref, alts, rec_types,
                                                       image_height=image_height, image_width=image_width)

            img_rec = str('\t'.join(str(item) for item in img_record))
            img_set.append(np.array(image_array, dtype=np.int8))
            smry.write(os.path.abspath(hdf5_filename) + ',' + str(indx) + ',' + img_rec + '\n')
            indx += 1

        img_dset = hdf5_file.create_dataset("images", (len(img_set),) + (image_height, image_width, 7), np.int8, compression='gzip')
        img_dset[...] = img_set

    def parse_region(self, start_position, end_position, thread_no):
        """
        Iterate through all the reads that fall in a region, find candidates, label candidates and output a bed file.
        :param start_position: Start position of the region
        :param end_position: End position of the region
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
        selected_candidates = candidate_finder.parse_reads_and_select_candidates(reads=reads)
        dictionaries_for_images = candidate_finder.get_pileup_dictionaries()

        if self.confident_tree is not None:
            confident_labeled = []
            for candidate in selected_candidates:
                pos_st = candidate[1] + 1# offset for bed
                pos_end = candidate[1] + 1# offset for bed
                in_conf = self.in_confident_check(pos_st, pos_end)
                if in_conf is True:
                    confident_labeled.append(candidate)
            selected_candidates = confident_labeled

        image_generator = ImageGenerator(dictionaries_for_images)

        if DEBUG_PRINT_CANDIDATES:
            for candidate in selected_candidates:
                print(candidate)

        self.generate_candidate_images(selected_candidates, image_generator, thread_no)


def parallel_run(chr_name, bam_file, ref_file, output_dir, start_pos, end_pos, conf_bed_tree, thread_no):
    """
    Run this method in parallel
    :param chr_name: Chromosome name
    :param bam_file: Bam file path
    :param ref_file: Ref file path
    :param output_dir: Output directory
    :param vcf_file: VCF file path
    :param start_position: Start position
    :param end_position: End position
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
    Create an internal directory inside the output directory to dump choromosomal bed files
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


def chromosome_level_parallelization(chr_name, bam_file, ref_file, output_path, max_threads, confident_bed_tree):
    """
    This method takes one chromosome name as parameter and chunks that chromosome in max_threads.
    :param chr_name: Chromosome name
    :param bam_file: Bam file
    :param ref_file: Ref file
    :param output_dir: Output directory
    :param max_threads: Maximum number of threads
    :return: A list of results returned by the processes
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
    :param bam_file: BAM file path
    :param ref_file: Reference file path
    :param output_dir_path: Output directory
    :param max_threads: Maximum number of threads to create in chromosome level
    :return: Saves a bed file
    """
    # chr_list = ["chr1", "chr2", "chr3", "chr4", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11",
    #             "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22"]
    # chr_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
    # "19"]

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
    Processes arguments and performs tasks to generate the pileup.
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
