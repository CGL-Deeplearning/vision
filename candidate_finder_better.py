import argparse
import math
import time
import csv
import os
import sys
import multiprocessing

from modules.core.CandidateFinder import CandidateFinder
from modules.handlers.BamHandler import BamHandler
from modules.handlers.FastaHandler import FastaHandler
from modules.handlers.TextColor import TextColor
from modules.handlers.FileManager import FileManager
from modules.core.IntervalTree import IntervalTree
from modules.handlers.TsvHandler import TsvHandler
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


def build_chromosomal_interval_trees(confident_bed_path):
    """
    Produce a dictionary of intervals trees, with one tree per chromosome
    :param confident_bed_path: Path to confident bed file
    :return: trees_chromosomal
    """
    tsv_handler_reference = TsvHandler(tsv_file_path=confident_bed_path)
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
    def __init__(self, chromosome_name, bam_file_path, reference_file_path, confident_tree, output_file_path):
        # --- initialize handlers ---
        self.bam_handler = BamHandler(bam_file_path)
        self.fasta_handler = FastaHandler(reference_file_path)
        self.output_dir = output_file_path
        self.confident_tree = confident_tree[chromosome_name]

        # --- initialize parameters ---
        self.chromosome_name = chromosome_name

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

    @staticmethod
    def write_bed(output_dir, chromosome_name, candidate_list):
        file_name = output_dir + chromosome_name + "_candidates" + ".bed"
        with open(file_name, 'w', newline='\n') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')
            for record in candidate_list:
                writer.writerow(record)

    def parse_region(self, start_position, end_position):
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

        if self.confident_tree is not None:
            confident_labeled = []
            for candidate in selected_candidates:
                pos_st = candidate[1]
                pos_end = candidate[2]
                in_conf = self.in_confident_check(pos_st, pos_end)
                if in_conf is True:
                    confident_labeled.append(candidate)
            selected_candidates = confident_labeled

        if DEBUG_PRINT_CANDIDATES:
            for candidate in selected_candidates:
                print(candidate)

        # labeled_sites = self.get_labeled_candidate_sites(selected_candidates, start_position, end_position, True)
        # self.write_bed(start_position, end_position, selected_candidates)
        return selected_candidates

    def test(self):
        """
        Run a test
        :return:
        """
        start_time = time.time()
        self.parse_region(start_position=100000, end_position=200000)
        end_time = time.time()
        print("TOTAL TIME ELAPSED: ", end_time-start_time)


def parallel_run(chr_name, bam_file, ref_file, confident_tree, output_dir, start_position, end_position):
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
                   confident_tree=confident_tree)

    # return the results
    candidate_list = view_ob.parse_region(start_position, end_position)
    return candidate_list


def chromosome_level_parallelization(chr_name, bam_file, ref_file, confident_tree, output_dir, max_threads):
    """
    This method takes one chromosome name as parameter and chunks that chromosome in max_threads.
    :param chr_name: Chromosome name
    :param bam_file: Bam file
    :param ref_file: Ref file
    :param output_dir: Output directory
    :param max_threads: Maximum number of threads
    :return: A list of results returned by the processes
    """
    # entire length of chromosome
    fasta_handler = FastaHandler(ref_file)
    whole_length = fasta_handler.get_chr_sequence_length(chr_name)

    # 2MB segments at once
    each_segment_length = 200000

    # chunk the chromosome into 1000 pieces
    chunks = int(math.ceil(whole_length / each_segment_length))
    args_list = []
    all_selected_candidates = []
    for i in range(chunks):
        # parse window of the segment. Use a 1000 overlap for corner cases.
        start_position = i * each_segment_length
        end_position = min((i + 1) * each_segment_length, whole_length)
        args = (chr_name, bam_file, ref_file, confident_tree, output_dir, start_position, end_position)
        args_list.append(args)
        if len(args_list) == max_threads:
            p = multiprocessing.Pool(processes=max_threads)
            data = p.starmap(parallel_run, args_list)
            p.close()
            data = [item for sublist in data for item in sublist]
            all_selected_candidates.extend(data)
            args_list = []
    if len(args_list):
        p = multiprocessing.Pool(processes=max_threads)
        data = p.starmap(parallel_run, args_list)
        p.close()
        data = [item for sublist in data for item in sublist]
        all_selected_candidates.extend(data)

    return all_selected_candidates


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

    return path_to_dir


def genome_level_parallelization(bam_file, ref_file, output_dir_path, max_threads):
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
    program_start_time = time.time()

    chr_list = ["chr19"]

    # each chormosome in list
    for chr in chr_list:
        sys.stderr.write(TextColor.BLUE + "STARTING " + str(chr) + " PROCESSES" + "\n")
        start_time = time.time()

        # create dump directory inside output directory
        output_dir = create_output_dir_for_chromosome(output_dir_path, chr)

        # do a chromosome level parallelization
        chromosome_level_parallelization(chr, bam_file, ref_file, output_dir, max_threads)

        end_time = time.time()
        sys.stderr.write(TextColor.PURPLE + "FINISHED " + str(chr) + " PROCESSES" + "\n")
        sys.stderr.write(TextColor.CYAN + "TIME ELAPSED: " + str(end_time - start_time) + "\n")

    # wait for the last process to end before file processing
    while True:
        if len(multiprocessing.active_children()) == 0:
            break

    for chr in chr_list:
        # here we dumped all the bed files
        path_to_dir = output_dir_path + chr + "/"

        concatenated_file_name = output_dir_path + chr + ".bed"

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
        chromosomal_tree = build_chromosomal_interval_trees(FLAGS.confident_bed)
    else:
        chromosomal_tree = None

    if FLAGS.test is True:
        view = View(chromosome_name=FLAGS.chromosome_name,
                    bam_file_path=FLAGS.bam,
                    reference_file_path=FLAGS.ref,
                    output_file_path=FLAGS.output_dir,
                    confident_tree=chromosomal_tree)
        view.test()
    elif FLAGS.chromosome_name is not None:
        all_candidates = chromosome_level_parallelization(FLAGS.chromosome_name, FLAGS.bam, FLAGS.ref,
                                                          chromosomal_tree, FLAGS.output_dir, FLAGS.max_threads)
        View.write_bed(FLAGS.output_dir, FLAGS.chromosome_name, all_candidates)
    else:
        genome_level_parallelization(FLAGS.bam, FLAGS.ref, FLAGS.output_dir, FLAGS.max_threads)
