import argparse
import math
import time
import os
import sys
import multiprocessing
from tqdm import tqdm

from modules.core.CandidateFinder import CandidateFinder
from modules.handlers.BamHandler import BamHandler
from modules.handlers.FastaHandler import FastaHandler
from modules.handlers.TextColor import TextColor
from modules.handlers.TsvHandler import TsvHandler
from modules.core.ImageGenerator import ImageGenerator
from modules.handlers.VcfHandler import VCFFileProcessor
from modules.core.CandidateLabelerHap import CandidateLabeler
from modules.handlers.FileManager import FileManager
from modules.core.LocalAssembler import LocalAssembler
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
    def __init__(self, chromosome_name, bam_file_path, reference_file_path, vcf_path, output_file_path, confident_tree):
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
        self.output_dir = output_file_path
        self.confident_tree = confident_tree[chromosome_name] if confident_tree else None
        self.vcf_handler = VCFFileProcessor(file_path=vcf_path)

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

    def get_labeled_candidate_sites(self, selected_candidate_list, start_pos, end_pos, filter_hom_ref=False):
        """
        Lable selected candidates of a region and return a list of records
        :param selected_candidate_list: List of all selected candidates with their alleles
        :param start_pos: start position of the region
        :param end_pos: end position of the region
        :param filter_hom_ref: whether to ignore hom_ref VCF records during candidate validation
        :return: labeled_sites: Labeled candidate sites. Each containing proper genotype.
        """
        # get dictionary of variant records for full region
        # self.vcf_handler.populate_dictionary(contig=self.chromosome_name,
        #                                      start_pos=start_pos - 5,
        #                                      end_pos=end_pos + 5,
        #                                      hom_filter=filter_hom_ref)

        # get separate positional variant dictionaries for IN, DEL, and SNP
        # positional_variants = self.vcf_handler.get_variant_dictionary()

        # create an object for labling the allele
        allele_labler = CandidateLabeler(fasta_handler=self.fasta_handler, vcf_handler=self.vcf_handler)
        # allele_labler = CandidateLabeler(fasta_handler=self.fasta_handler)

        # label the sites
        # labeled_sites = allele_labler.get_labeled_candidates(chromosome_name=self.chromosome_name,
        #                                                      positional_vcf=positional_variants,
        #                                                      candidate_sites=selected_candidate_list)
        labeled_sites = allele_labler.get_labeled_candidates(chromosome_name=self.chromosome_name,
                                                             pos_start=start_pos,
                                                             pos_end=end_pos,
                                                             candidate_sites=selected_candidate_list)

        return labeled_sites

    def parse_region(self, start_position, end_position, thread_no):
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
        labeled_sites = self.get_labeled_candidate_sites(selected_candidates, start_position, end_position, True)

        # create image generator object with all necessary dictionary
        image_generator = ImageGenerator(dictionaries_for_images)

        if DEBUG_PRINT_CANDIDATES:
            for candidate in labeled_sites:
                print(candidate)

        # generate and save candidate images
        ImageGenerator.generate_and_save_candidate_images(self.chromosome_name, labeled_sites, image_generator,
                                                          thread_no, self.output_dir,
                                                          image_height=100, image_width=222, image_channels=6)

        # end_time = time.time()
        # print("TIME ELAPSED: ", start_position, end_position, end_time - st_time)


def parallel_run(chr_name, bam_file, ref_file, vcf_file, output_dir, start_pos, end_pos, conf_bed_tree, thread_no):
    """
    Creates a view object for a region and generates images for that region.
    :param chr_name: Name of the chromosome
    :param bam_file: path to BAM file
    :param ref_file: path to reference FASTA file
    :param vcf_file: path to VCF file
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
                   vcf_path=vcf_file,
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


def chromosome_level_parallelization(chr_name, bam_file, ref_file, vcf_file, output_path, max_threads,
                                     confident_bed_tree, singleton_run=False):
    """
    This method takes one chromosome name as parameter and chunks that chromosome in max_threads.
    :param chr_name: Name of the chromosome
    :param bam_file: path to BAM file
    :param ref_file: path to reference FASTA file
    :param vcf_file: path to VCF file
    :param output_path: path to output directory
    :param max_threads: Maximum number of threads to run at one instance
    :param confident_bed_tree: tree containing confident bed intervals
    :param singleton_run: if running a chromosome independently
    :return:
    """
    sys.stderr.write(TextColor.BLUE + "STARTING " + str(chr_name) + " PROCESSES" + "\n" + TextColor.END)
    # create dump directory inside output directory
    output_dir = create_output_dir_for_chromosome(output_path, chr_name)

    intervals = []
    # if there's no confident bed provided, then chop the chromosome
    if confident_bed_tree is None:
        # entire length of chromosome
        fasta_handler = FastaHandler(ref_file)
        whole_length = fasta_handler.get_chr_sequence_length(chr_name)
        # .5MB segments at once
        each_segment_length = 1000

        # chunk the chromosome into pieces
        chunks = int(math.ceil(whole_length / each_segment_length))

        for i in tqdm(range(chunks), ncols=100):
            start_position = i * each_segment_length
            end_position = min((i + 1) * each_segment_length, whole_length)
            intervals.append([start_position, end_position])
    else:
        each_segment_length = 1000
        for interval in confident_bed_tree[chr_name]:
            start_position, end_position = interval
            start_position = start_position - BED_POSITION_BUFFER
            end_position = end_position + BED_POSITION_BUFFER
            if end_position - start_position + 1 < each_segment_length:
                intervals.append([start_position, end_position])
            else:
                st_ = start_position
                while st_ < end_position:
                    end_ = min(st_ + each_segment_length, end_position)
                    intervals.append([st_, end_])
                    st_ = end_ + 1

    for i in tqdm(range(len(intervals)), ncols=100):
        start_position = intervals[i][0]
        end_position = intervals[i][1]
        # gather all parameters
        args = (chr_name, bam_file, ref_file, vcf_file, output_dir, start_position, end_position, confident_bed_tree, i)

        p = multiprocessing.Process(target=parallel_run, args=args)
        p.start()

        # wait until we have room for new processes to start
        while True:
            if len(multiprocessing.active_children()) < max_threads:
                break

    if singleton_run:
        # wait for the last process to end before file processing
        while True:
            if len(multiprocessing.active_children()) == 0:
                break
        # remove summary files and make one file
        summary_file_to_csv(output_path, [chr_name])


def genome_level_parallelization(bam_file, ref_file, vcf_file, output_dir_path, max_threads, confident_bed_tree):
    """
    This method calls chromosome_level_parallelization for each chromosome.
    :param bam_file: path to BAM file
    :param ref_file: path to reference FASTA file
    :param vcf_file: path to VCF file
    :param output_dir_path: path to output directory
    :param max_threads: Maximum number of threads to run at one instance
    :param confident_bed_tree: tree containing confident bed intervals
    :return:
    """

    # --- NEED WORK HERE --- GET THE CHROMOSOME NAMES FROM THE BAM FILE
    # chr_list = ["chr20", "chr21", "chr22"]
    chr_list = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12",
                "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19"]

    program_start_time = time.time()

    # chr_list = ["19"]

    # each chromosome in list
    for chr_name in chr_list:

        start_time = time.time()

        # do a chromosome level parallelization
        chromosome_level_parallelization(chr_name, bam_file, ref_file, vcf_file, output_dir_path,
                                         max_threads, confident_bed_tree)

        end_time = time.time()
        sys.stderr.write(TextColor.PURPLE + "FINISHED " + str(chr_name) + " PROCESSES" + "\n")
        sys.stderr.write(TextColor.CYAN + "TIME ELAPSED: " + str(end_time - start_time) + "\n")

    # wait for the last process to end before file processing
    while True:
        if len(multiprocessing.active_children()) == 0:
            break

    summary_file_to_csv(output_dir_path, chr_list)

    program_end_time = time.time()
    sys.stderr.write(TextColor.RED + "PROCESSED FINISHED SUCCESSFULLY" + "\n")
    sys.stderr.write(TextColor.CYAN + "TOTAL TIME FOR GENERATING ALL RESULTS: " + str(program_end_time-program_start_time) + "\n")


def summary_file_to_csv(output_dir_path, chr_list):
    """
    Remove the abundant number of summary files and bind them to one
    :param output_dir_path: Path to the output directory
    :param chr_list: List of chromosomes
    :return:
    """
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


def test(view_object):
    """
    Run a test
    :return:
    """
    start_time = time.time()
    # view_object.parse_region(start_position=8926678, end_position=8927210, thread_no=1)
    view_object.parse_region(start_position=8926678, end_position=8927210, thread_no=1)
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
        "--vcf",
        type=str,
        required=True,
        help="VCF file path."
    )
    parser.add_argument(
        "--confident_bed",
        type=str,
        default='',
        help="Path to confident BED file"
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
    if FLAGS.confident_bed != '' and FLAGS.test is False:
        confident_intervals = View.build_chromosomal_interval_trees(FLAGS.confident_bed)
    else:
        confident_intervals = None

    if confident_intervals is not None:
        sys.stderr.write(TextColor.PURPLE + "CONFIDENT TREE LOADED\n" + TextColor.END)
    else:
        sys.stderr.write(TextColor.RED + "CONFIDENT BED IS NULL\n" + TextColor.END)

    if FLAGS.test is True:
        chromosome_output = create_output_dir_for_chromosome(FLAGS.output_dir, FLAGS.chromosome_name)
        view = View(chromosome_name=FLAGS.chromosome_name,
                    bam_file_path=FLAGS.bam,
                    reference_file_path=FLAGS.ref,
                    vcf_path=FLAGS.vcf,
                    output_file_path=chromosome_output,
                    confident_tree=confident_intervals)
        test(view)
    elif FLAGS.chromosome_name is not None:
        chromosome_level_parallelization(FLAGS.chromosome_name, FLAGS.bam, FLAGS.ref, FLAGS.vcf, FLAGS.output_dir,
                                         FLAGS.max_threads, confident_intervals, singleton_run=True)
    else:
        genome_level_parallelization(FLAGS.bam, FLAGS.ref, FLAGS.vcf, FLAGS.output_dir,
                                     FLAGS.max_threads, confident_intervals)
