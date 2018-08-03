from modules.core.AlignGraphCandidateFinder import CandidateFinder
from modules.core.AlignmentGraph import AlignmentGraph
from modules.core.AlignmentGraphLabeler import AlignmentGraphLabeler
from modules.core.IntervalTree import IntervalTree
from modules.core.IterativeHistogram import IterativeHistogram
from modules.handlers.FastaHandler import FastaHandler
from modules.handlers.VcfHandler import VCFFileProcessor
from modules.handlers.BamHandler import BamHandler
from modules.handlers.TsvHandler import TsvHandler
from modules.handlers.FileManager import FileManager
from matplotlib import pyplot
from multiprocessing import Pool
import multiprocessing
import numpy
import sys
import os
import csv

REF, SNP, INS, DEL = 0, 1, 2, 3
cigar_code_names = ["REF","SNP","INS","DEL"]


def build_chromosomal_interval_trees(confident_bed_path):
    """
    Produce a dictionary of intervals trees, with one tree per chromosome
    :param confident_bed_path: Path to confident bed file
    :return: trees_chromosomal
    """
    # create an object for tsv file handling
    tsv_handler_reference = TsvHandler(tsv_file_path=confident_bed_path)
    # create intervals based on chromosome
    intervals_chromosomal_reference = tsv_handler_reference.get_bed_intervals_by_chromosome(start_offset=1,
                                                                                            universal_offset=-1)
    # create a dictionary to get all chromosomal trees
    trees_chromosomal = dict()

    # for each chromosome extract the tree and add it to the dictionary
    for chromosome_name in intervals_chromosomal_reference:
        intervals = intervals_chromosomal_reference[chromosome_name]
        tree = IntervalTree(intervals)

        trees_chromosomal[chromosome_name] = tree

    # return the dictionary containing all the trees
    return trees_chromosomal


def get_node_representation_data(reference_file_path, bam_file_path, vcf_path, chromosome_name, start_position, end_position):
    fasta_handler = FastaHandler(reference_file_path)
    bam_handler = BamHandler(bam_file_path)
    vcf_handler = VCFFileProcessor(vcf_path)

    reference_sequence = fasta_handler.get_sequence(chromosome_name=chromosome_name,
                                                    start=start_position,
                                                    stop=end_position+1)

    alignment_graph = AlignmentGraph(chromosome_name=chromosome_name,
                                     start_position=start_position,
                                     end_position=end_position)

    # get the reads that fall in that region
    reads = bam_handler.get_reads(chromosome_name=chromosome_name,
                                  start=start_position,
                                  stop=end_position)

    vcf_handler.populate_dictionary(contig=chromosome_name,
                                    start_pos=start_position,
                                    end_pos=end_position+1)

    positional_variants = vcf_handler.get_variant_dictionary()

    # create candidate finder object
    graph_candidate_finder = CandidateFinder(reads=reads,
                                             fasta_handler=fasta_handler,
                                             chromosome_name=chromosome_name,
                                             region_start_position=start_position,
                                             region_end_position=end_position,
                                             alignment_graph=alignment_graph)

    total_reads = graph_candidate_finder.get_read_alignment_data(reads=reads)

    if total_reads == 0:
        return

    labeler = AlignmentGraphLabeler(reference_sequence=reference_sequence,
                                    positional_variants=positional_variants,
                                    chromosome_name=chromosome_name,
                                    start_position=start_position,
                                    end_position=end_position,
                                    graph=alignment_graph)

    labeler.parse_region()

    node_counts = labeler.get_node_counts()
    false_negative_positions = labeler.get_false_negative_positions()

    # print(false_negative_positions)

    return node_counts, false_negative_positions


def parallel_run(reference_file_path, bed_file_path, vcf_file_path, bam_file_path, chromosome_name, start_position, end_position, return_dict, counter, n_chunks):
    chromosomal_interval_tree = build_chromosomal_interval_trees(bed_file_path)

    vcf_handler = VCFFileProcessor(vcf_file_path)
    vcf_handler.populate_dictionary(contig=chromosome_name,
                                    start_pos=start_position,
                                    end_pos=end_position)
    positional_variants = vcf_handler.get_variant_dictionary()

    all_node_representation = [[0,0] for cigar in [REF, SNP, INS, DEL]]
    all_false_negative_positions = list()

    for p,position in enumerate(positional_variants):
        interval = [position,position]
        if interval not in chromosomal_interval_tree[chromosome_name]:
            continue

        start_position = position - 2
        end_position = position + 12

        # get stats on nodes
        node_representation, \
        false_negative_positions = get_node_representation_data(reference_file_path=reference_file_path,
                                                                bam_file_path=bam_file_path,
                                                                vcf_path=vcf_file_path,
                                                                chromosome_name=chromosome_name,
                                                                start_position=start_position,
                                                                end_position=end_position)

        all_false_negative_positions.extend(false_negative_positions)

        for cigar_code in [REF,SNP,INS,DEL]:
            for status in [0,1]:
                all_node_representation[cigar_code][status] += node_representation[cigar_code][status]

    counter.value += 1
    sys.stdout.write('\r' + "%.2f%% Completed"%(100*counter.value/n_chunks))

    return_dict[start_position] = [all_node_representation, all_false_negative_positions]

    return


def iterate_datasets():
    # ---- GIAB (dev machine) -------------------------------------------------
    # bam_file_path = "/home/ryan/data/GIAB/NA12878_GIAB_30x_GRCh37.sorted.bam"
    # reference_file_path = "/home/ryan/data/GIAB/GRCh37_WG.fa"
    # vcf_file_path = "/home/ryan/data/GIAB/NA12878_GRCh37.vcf.gz"
    # bed_file_path = "/home/ryan/data/GIAB/NA12878_GRCh37_confident_chr18.bed"
    #
    # chromosome_name = "18"
    # output_dir = "node_stats/GIAB/"
    #
    # test_with_BAM_data(reference_file_path, bed_file_path, vcf_file_path, bam_file_path, chromosome_name, output_dir)

    # ---- Nanopore ALBACORE? (dev machine) -----------------------------------
    bam_file_path = "/home/ryan/data/Nanopore/BAM/Albacore/chr18.sorted.bam"
    reference_file_path = "/home/ryan/data/GIAB/GRCh38_WG.fa"
    vcf_file_path = "/home/ryan/data/GIAB/NA12878_GRCh38_PG.vcf.gz"
    bed_file_path = "/home/ryan/data/GIAB/NA12878_GRCh38_confident.bed"

    chromosome_name = "chr18"
    output_dir = "node_stats/rel4/"

    test_with_BAM_data(reference_file_path, bed_file_path, vcf_file_path, bam_file_path, chromosome_name, output_dir)

    # ---- Nanopore GUPPY (dev machine) ---------------------------------------
    bam_file_path = "/home/ryan/data/Nanopore/BAM/Guppie/rel5-guppy-0.3.0-chunk10k.sorted.bam"
    reference_file_path = "/home/ryan/data/GIAB/GRCh38_WG.fa"
    vcf_file_path = "/home/ryan/data/GIAB/NA12878_GRCh38_PG.vcf.gz"
    bed_file_path = "/home/ryan/data/GIAB/NA12878_GRCh38_confident.bed"

    chromosome_name = "chr18"
    output_dir = "node_stats/rel5/"

    test_with_BAM_data(reference_file_path, bed_file_path, vcf_file_path, bam_file_path, chromosome_name, output_dir)

    # -------------------------------------------------------------------------


def parse_thread_dictionary(dictionary, output_dir):
    """
    For a dictionary of type Multiprocessing.Manager.dict, take each thread's return data and combine into a single
    dataset for as many datasets that exist in each thread's output
    :param dictionary:
    :return:
    """
    all_node_representation = [[0,0] for cigar in [REF, SNP, INS, DEL]]
    all_false_negative_positions = list()

    file = open(os.path.join(output_dir, "BAM_representation_report.tsv"), 'w')
    writer = csv.writer(file, delimiter='\t')

    for key in dictionary.keys():
        node_representation, false_negative_positions = dictionary[key]
        all_false_negative_positions.extend(false_negative_positions)

        for cigar_code in [REF, SNP, INS, DEL]:
            for status in [0, 1]:
                all_node_representation[cigar_code][status] += node_representation[cigar_code][status]

    # print node representation
    for cigar_code in [REF, SNP, INS, DEL]:
        for status in [0, 1]:
            row = [cigar_code_names[cigar_code], status, all_node_representation[cigar_code][status]]
            writer.writerow(row)

    file.write('\n')

    for false_negative in all_false_negative_positions:
        position, cigar_code, sequence = false_negative
        row = [position, cigar_code_names[cigar_code], sequence]

        writer.writerow(row)
    return


def test_with_BAM_data(reference_file_path, bed_file_path, vcf_file_path, bam_file_path, chromosome_name, output_dir):
    """
    :param reference_file_path:
    :param bed_file_path:
    :param vcf_file_path:
    :param bam_file_path:
    :param chromosome_name:
    :return:
    """
    FileManager.ensure_directory_exists(output_dir)

    if sys.platform == 'win32':
        available_threads = int(os.environ['NUMBER_OF_PROCESSORS'])
    else:
        available_threads = int(os.popen('grep -c cores /proc/cpuinfo').read())

    # max_threads = available_threads - 4
    max_threads = 1

    start_position = 60080000
    end_position = 60090000
    chunk_size = 25000

    steps = range(start_position, end_position + chunk_size, chunk_size)
    n_chunks = len(steps)

    print(list(steps))

    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)

    return_dict = manager.dict()

    # get entire set of files
    args = list()

    # generate exhaustive list of arguments to be sent to each thread
    for i in range(len(steps)-1):
        start_chunk = steps[i]
        end_chunk = steps[i+1]

        args.append((reference_file_path,
                     bed_file_path,
                     vcf_file_path,
                     bam_file_path,
                     chromosome_name,
                     start_chunk,
                     end_chunk,
                     return_dict,
                     counter,
                     n_chunks))

    # initiate threading
    with Pool(processes=max_threads) as pool:
        pool.starmap(parallel_run, args)

    print()

    parse_thread_dictionary(return_dict, output_dir)
    return


if __name__ == "__main__":
    iterate_datasets()
