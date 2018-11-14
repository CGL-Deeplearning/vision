import argparse
import math
import time
import os
import sys
import h5py
import numpy as np

from modules.handlers.IntervalTree import IntervalTree
from modules.core.CandidateFinder import CandidateFinder
from modules.handlers.BamHandler import BamHandler
from modules.handlers.FastaHandler import FastaHandler
from modules.handlers.TextColor import TextColor
from modules.handlers.TsvHandler import TsvHandler
from modules.core.ImageGenerator import ImageGenerator
from modules.handlers.VcfHandler import VCFFileProcessor
from modules.core.CandidateLabelerHap import CandidateLabeler
from modules.core.LocalAssembler import LocalAssembler
from modules.core.OptionValues import ImageOptions
"""
This script creates training images from BAM, Reference FASTA and truth VCF file. The process is:
- Find candidates that can be variants
- Label candidates using the VCF
- Create images for each candidate

Input:
- BAM file: Alignment of a genome
- REF file: The reference FASTA file used in the alignment
- VCF file: A truth VCF file
- BED file: A confident bed file. If confident_bed is passed it will only generate train set for those region.

Output:
- H5PY files: Containing images and their label of the genome.
- CSV file: Containing records of images and their location in the H5PY file.
"""

# Global debug helpers
DEBUG_PRINT_CANDIDATES = False
DEBUG_TIME_PROFILE = False
DEBUG_TEST_PARALLEL = False
BED_POSITION_BUFFER = 0


class View:
    """
    Process manager that runs sequence of processes to generate images and their labebls.
    """
    def __init__(self, chromosome_name, bam_file_path, reference_file_path, vcf_path, confident_tree, train_mode):
        """
        Initialize a manager object
        :param chromosome_name: Name of the chromosome
        :param bam_file_path: Path to the BAM file
        :param reference_file_path: Path to the reference FASTA file
        :param vcf_path: Path to the VCF file
        :param output_file_path: Path to the output directory where images are saved
        :param confident_tree: Dictionary containing all confident trees. NULL if parameter not passed.
        """
        # --- initialize handlers ---
        # create objects to handle different files and query
        self.bam_handler = BamHandler(bam_file_path)
        self.fasta_handler = FastaHandler(reference_file_path)
        self.confident_tree = confident_tree[chromosome_name] if confident_tree else None
        self.interval_tree = IntervalTree(self.confident_tree)
        self.vcf_handler = VCFFileProcessor(file_path=vcf_path) if train_mode else None

        # --- initialize names ---
        # name of the chromosome
        self.chromosome_name = chromosome_name

    @staticmethod
    def build_chromosomal_interval_trees(confident_bed_path):
        """
        Produce a dictionary of intervals trees, with one tree per chromosome
        :param confident_bed_path: Path to confident bed file
        :return: trees_chromosomal
        """
        # create an object for tsv file handling
        tsv_handler_reference = TsvHandler(tsv_file_path=confident_bed_path)
        # create intervals based on chromosome
        intervals_chromosomal_reference = tsv_handler_reference.get_bed_intervals_by_chromosome(universal_offset=-1)

        return intervals_chromosomal_reference

    def get_labeled_candidate_sites(self, selected_candidate_list, start_pos, end_pos):
        """
        Lable selected candidates of a region and return a list of records
        :param selected_candidate_list: List of all selected candidates with their alleles
        :param start_pos: start position of the region
        :param end_pos: end position of the region
        :param filter_hom_ref: whether to ignore hom_ref VCF records during candidate validation
        :return: labeled_sites: Labeled candidate sites. Each containing proper genotype.
        """
        # create an object for labling the allele
        allele_labler = CandidateLabeler(fasta_handler=self.fasta_handler, vcf_handler=self.vcf_handler)

        labeled_sites = allele_labler.get_labeled_candidates(chromosome_name=self.chromosome_name,
                                                             pos_start=start_pos,
                                                             pos_end=end_pos,
                                                             candidate_sites=selected_candidate_list)

        return labeled_sites

    @staticmethod
    def overlap_length_between_ranges(range_a, range_b):
        return max(0, (min(range_a[1], range_b[1]) - max(range_a[0], range_b[0])))

    def parse_region(self, start_position, end_position, label_candidates):
        """
        Generate labeled images of a given region of the genome
        :param start_position: Start position of the region
        :param end_position: End position of the region
        :param thread_no: Thread no for this region
        :return:
        """
        # st_time = time.time()
        local_assembler = LocalAssembler(self.fasta_handler,
                                         self.bam_handler,
                                         self.chromosome_name,
                                         start_position,
                                         end_position)
        reads = local_assembler.perform_local_assembly(perform_alignment=True)

        # create candidate finder object
        candidate_finder = CandidateFinder(fasta_handler=self.fasta_handler,
                                           chromosome_name=self.chromosome_name,
                                           region_start_position=start_position,
                                           region_end_position=end_position)

        # go through each read and find candidate positions and alleles
        selected_candidates = candidate_finder.parse_reads_and_select_candidates(reads=reads)
        dictionaries_for_images = candidate_finder.get_pileup_dictionaries()

        # get all labeled candidate sites
        if label_candidates:
            confident_intervals_in_region = self.interval_tree.find(start_position, end_position)
            if not confident_intervals_in_region:
                return None, None, None

            confident_candidates = []
            for candidate in selected_candidates:
                for interval in confident_intervals_in_region:
                    if self.overlap_length_between_ranges((candidate[1], candidate[2]), interval) > 0:
                        confident_candidates.append(candidate)
                        break

            if not confident_candidates:
                return None, None, None

            labeled_sites = self.get_labeled_candidate_sites(confident_candidates, start_position, end_position)
        else:
            labeled_sites = selected_candidates

        # create image generator object with all necessary dictionary
        image_generator = ImageGenerator(dictionaries_for_images)

        # generate and save candidate images
        img_set, label_set, img_recs = ImageGenerator.generate_and_save_candidate_images(labeled_sites,
                                                                                         image_generator,
                                                                                         image_height=ImageOptions.IMAGE_HEIGHT,
                                                                                         image_width=ImageOptions.IMAGE_WIDTH,
                                                                                         train_mode=label_candidates)
        return img_set, label_set, img_recs


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



def chromosome_level_parallelization(chr_list,
                                     bam_file,
                                     ref_file,
                                     vcf_file,
                                     confident_intervals,
                                     output_path,
                                     image_path,
                                     total_threads,
                                     thread_id,
                                     train_mode,
                                     max_size=1000):
    """
    This method takes one chromosome name as parameter and chunks that chromosome in max_threads.
    :param chr_list: List of chromosomes to be processed
    :param bam_file: path to BAM file
    :param ref_file: path to reference FASTA file
    :param vcf_file: path to VCF file
    :param max_size: Maximum size of a segment
    :param output_path: path to output directory
    :return:
    """
    # if there's no confident bed provided, then chop the chromosome
    fasta_handler = FastaHandler(ref_file)

    for chr_name in chr_list:
        # interval_start, interval_end = (0, fasta_handler.get_chr_sequence_length(chr_name) + 1)
        interval_start, interval_end = (2005000, 2010000)
        # interval_start, interval_end = (269856, 269996)
        # interval_start, interval_end = (1413980, 1413995)
        # interval_start, interval_end = (260000, 260999)

        all_intervals = []
        for pos in range(interval_start, interval_end, max_size):
            all_intervals.append((pos, min(interval_end, pos + max_size - 1)))

        intervals = [r for i, r in enumerate(all_intervals) if i % total_threads == thread_id]

        view = View(chromosome_name=chr_name,
                    bam_file_path=bam_file,
                    reference_file_path=ref_file,
                    vcf_path=vcf_file,
                    confident_tree=confident_intervals,
                    train_mode=train_mode)

        smry = None
        image_file_name = image_path + chr_name + "_" + str(thread_id) + ".h5py"
        if intervals:
            smry = open(output_path + chr_name + "_" + str(thread_id) + "_summary.csv", 'w')

        start_time = time.time()
        total_reads_processed = 0
        total_windows = 0
        all_images = []
        all_labels = []
        global_index = 0
        for interval in intervals:
            _start, _end = interval
            img_set, label_set, img_recs = view.parse_region(start_position=_start,
                                                             end_position=_end,
                                                             label_candidates=train_mode)

            if not img_set or not img_recs:
                continue

            # save the images
            for i, image in enumerate(img_set):
                all_images.append(image)
                if train_mode:
                    all_labels.append(label_set[i])

                smry.write(os.path.abspath(image_file_name) + ',' + str(global_index) + ',' + img_recs[i] + '\n')
                global_index += 1

        hdf5_file = h5py.File(image_file_name, mode='w')
        # the image dataset we save. The index name in h5py is "images".
        img_dset = hdf5_file.create_dataset("images", (len(all_images),) + (ImageOptions.IMAGE_HEIGHT,
                                                                            ImageOptions.IMAGE_WIDTH,
                                                                            ImageOptions.IMAGE_CHANNELS), np.uint8,
                                            compression='gzip')
        label_dataset = hdf5_file.create_dataset("labels", (len(all_labels),), np.uint8)
        # save the images and labels to the h5py file
        img_dset[...] = all_images
        label_dataset[...] = all_labels
        hdf5_file.close()

        print("CHROMOSOME: ", chr_name,
              "THREAD ID: ", thread_id,
              "READS: ", total_reads_processed,
              "WINDOWS: ", total_windows,
              "TOTAL TIME ELAPSED: ", int(math.floor(time.time()-start_time)/60), "MINS",
              math.ceil(time.time()-start_time) % 60, "SEC")


def handle_output_directory(output_dir, thread_id):
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

    internal_directory = "images_" + str(thread_id) + "/"
    image_dir = output_dir + internal_directory

    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    return output_dir, image_dir


def boolean_string(s):
    """
    https://stackoverflow.com/questions/44561722/why-in-argparse-a-true-is-always-true
    :param s:
    :return:
    """
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'


def get_chromosme_list(chromosome_names):
    split_names = chromosome_names.strip().split(',')
    split_names = [name.strip() for name in split_names]

    chromosome_name_list = []
    for name in split_names:
        range_split = name.split('-')
        if len(range_split) > 1:
            chr_prefix = ''
            for p in name:
                if p.isdigit():
                    break
                else:
                    chr_prefix = chr_prefix + p

            int_ranges = []
            for item in range_split:
                s = ''.join(i for i in item if i.isdigit())
                int_ranges.append(int(s))
            int_ranges = sorted(int_ranges)

            for chr_seq in range(int_ranges[0], int_ranges[-1] + 1):
                chromosome_name_list.append(chr_prefix + str(chr_seq))
        else:
            chromosome_name_list.append(name)

    return chromosome_name_list


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bam",
        type=str,
        required=True,
        help="BAM file containing reads of interest."
    )
    parser.add_argument(
        "--fasta",
        type=str,
        required=True,
        help="Reference corresponding to the BAM file."
    )
    parser.add_argument(
        "--vcf",
        type=str,
        default=None,
        help="VCF file path."
    )
    parser.add_argument(
        "--bed",
        type=str,
        default=None,
        help="Path to confident BED file"
    )
    parser.add_argument(
        "--chromosome_name",
        type=str,
        help="Desired chromosome number E.g.: 3"
    )
    parser.add_argument(
        "--train_mode",
        type=boolean_string,
        default=False,
        help="If true then a dry test is run."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="candidate_finder_output/",
        help="Path to output directory."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=5,
        help="Number of maximum threads for this region."
    )
    parser.add_argument(
        "--thread_id",
        type=int,
        required=False,
        help="Reference corresponding to the BAM file."
    )
    FLAGS, unparsed = parser.parse_known_args()
    chr_list = get_chromosme_list(FLAGS.chromosome_name)
    # if the confident bed is not empty then create the tree
    if FLAGS.bed:
        confident_intervals = View.build_chromosomal_interval_trees(FLAGS.bed)
    else:
        confident_intervals = None

    if FLAGS.train_mode and (not confident_intervals or not FLAGS.vcf):
        sys.stderr.write(TextColor.RED + "ERROR: TRAIN MODE REQUIRES --vcf AND --bed TO BE SET.\n" + TextColor.END)
        exit(1)
    output_dir, image_dir = handle_output_directory(os.path.abspath(FLAGS.output_dir), FLAGS.thread_id)

    chromosome_level_parallelization(chr_list,
                                     FLAGS.bam,
                                     FLAGS.fasta,
                                     FLAGS.vcf,
                                     confident_intervals,
                                     output_dir,
                                     image_dir,
                                     FLAGS.threads,
                                     FLAGS.thread_id,
                                     FLAGS.train_mode)
