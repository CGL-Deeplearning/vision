from modules.handlers.VcfHandler import VCFFileProcessor
from modules.handlers.FastaHandler import FastaHandler
from modules.handlers.FileManager import FileManager
from modules.core.IntervalTree import IntervalTree
from modules.handlers.TsvHandler import TsvHandler
import numpy
import os.path
from tqdm import tqdm
import sys

SNP, IN, DEL = 0, 1, 2
VCF_OFFSET = 1


def get_positional_frequency_data(coordinates, frequency_data):
    """
    Convert a numpy array of coordinate based data into a dictionary with coordinates as keys
    :param coordinates:
    :param frequency_data:
    :return:
    """
    length = coordinates.shape[0]

    positional_frequency_data = dict()
    for i in range(length):
        start_position = int(coordinates[i][1])

        positional_frequency_data[start_position] = frequency_data[i,:]

    return positional_frequency_data


# def get_positional_support(coordinates, frequency_data):
#     """
#     Convert a numpy array of coordinate based data into a dictionary with coordinates as keys
#     :param coordinates:
#     :param frequency_data:
#     :return:
#     """
#
#     # find the number of bins for each frequency vector, for each type of allele
#     frequency_data_length = frequency_data.shape[0]
#     n_alleles_per_type = int(frequency_data_length/3)
#
#     snp_freq = frequency_data[:,n_alleles_per_type]
#     in_freq = frequency_data[:,n_alleles_per_type:n_alleles_per_type*2]
#     del_freq = frequency_data[:,n_alleles_per_type*2:]
#
#     # assume there is support for each type if the frequency of alternates is nonzero
#     snp_supported = (numpy.sum(snp_freq, axis=1) > 0)
#     in_supported = (numpy.sum(in_freq, axis=1) > 0)
#     del_supported = (numpy.sum(del_freq, axis=1) > 0)
#
#     support = numpy.concatenate([snp_supported, in_supported, del_supported], axis=1)
#
#     print(numpy.array2string(support[1:5,:], threshold=numpy.nan, separator="\t", precision=2, max_line_width=500))
#
#     length = coordinates.shape[0]
#
#     positional_support = dict()
#     for i in range(length):
#         start_position = int(coordinates[i][1])
#
#         positional_support[start_position] = support[i, :]
#
#     return positional_support

def generate_interval_tree_from_bed_file(regions_bed_path):
    tsv_handler = TsvHandler(regions_bed_path)

    # collect intervals from BED in illumina PG standards and convert to intervals that make sense: 0-based, closed
    bed_intervals_by_chromosome = tsv_handler.get_bed_intervals_by_chromosome(universal_offset=-1, start_offset=1)

    interval_trees_by_chromosome = dict()

    for chromosome in bed_intervals_by_chromosome:
        intervals = bed_intervals_by_chromosome[chromosome]

        interval_tree = IntervalTree(intervals)
        interval_trees_by_chromosome[chromosome] = interval_tree

    print("chromosomes: ", bed_intervals_by_chromosome.keys())

    return interval_trees_by_chromosome


def get_support(frequency_data):
    """
    Test whether there are any alternate alleles of type SNP/IN/DEL in a coordinate
    :param frequency_data:
    :return:
    """
    # find the number of bins for each frequency vector, for each type of allele
    frequency_data_length = frequency_data.shape[0]
    n_alleles_per_type = int(frequency_data_length/3)

    snp_freq = frequency_data[:n_alleles_per_type]
    in_freq = frequency_data[n_alleles_per_type:n_alleles_per_type*2]
    del_freq = frequency_data[n_alleles_per_type*2:]

    # assume there is support for each type if the frequency of alternates is nonzero
    snp_supported = (numpy.sum(snp_freq) > 0)
    in_supported = (numpy.sum(in_freq) > 0)
    del_supported = (numpy.sum(del_freq) > 0)

    support = [snp_supported, in_supported, del_supported]

    return support


def validate_positional_variants(positional_variants, positional_frequency_data, confident_interval_tree, start, stop):
    """
    Iterate vcf and candidate dictionaries to see whether candidates exist for each variant
    :param positional_variants:
    :param positional_frequency_data:
    :return:
    """
    false_negatives = list()
    n_false_negative = 0
    n_true_positive = 0

    # for all positions covered by this chunk
    for position in range(start+VCF_OFFSET, stop+VCF_OFFSET):
        # if there is a variant in this position
        if position in positional_variants and [position,position] in confident_interval_tree:
            # get variant records
            records = positional_variants[position]
            adjusted_position = (position - VCF_OFFSET)

            # if there are candidates, get determine their support by variant type
            if adjusted_position in positional_frequency_data:
                support = get_support(positional_frequency_data[adjusted_position])
            else:
                support = [False, False, False]

            # for each possible variant type
            for type_index in [SNP, IN, DEL]:
                # if there is a variant of this type
                if len(records[type_index]) > 0:
                    # test if there are candidates of that type
                    if not support[type_index]:
                        n_false_negative += 1
                        false_negatives.append([position, type_index, support, records[type_index]])
                    else:
                        n_true_positive += 1

    return n_false_negative, n_true_positive, false_negatives


def get_region_from_file_path(file_path):
    basename = os.path.basename(file_path)
    basename = basename.split(".npz")[0]
    tokens = basename.split('_')

    chromosome, start, stop = tokens[-3:]
    start = int(start)
    stop = int(stop)

    return chromosome, start, stop


def get_chromosome_lengths(chromosome_names, fasta_handler):
    chromosome_lengths = dict()
    for name in chromosome_names:
        length = fasta_handler.get_chr_sequence_length(chromosome_name=name)
        chromosome_lengths[name] = length

    return chromosome_lengths


def validate_regional_candidate_data(file_paths, vcf_path, fasta_path, bed_path):
    """
    Parse a table of candidate data to determine whether candidates have been found for every variant in their region
    :return:
    """
    total_false_negative = 0
    total_true_positive = 0
    all_false_negatives = list()
    vcf_handler = VCFFileProcessor(vcf_path)
    fasta_handler = FastaHandler(fasta_path)
    interval_trees_by_chromosome = generate_interval_tree_from_bed_file(bed_path)
    chromosome_names = fasta_handler.get_sequence_names()
    chromosome_lengths = get_chromosome_lengths(chromosome_names=chromosome_names, fasta_handler=fasta_handler)

    current_chromosome_name = None
    positional_variants = None
    i=0
    for path in tqdm(file_paths):
        i+=1
        data = numpy.load(path)['a'].T

        chromosome_name, start, stop = get_region_from_file_path(path)

        if current_chromosome_name != chromosome_name:
            print("current:", current_chromosome_name, "actual:", chromosome_name)

            vcf_handler.populate_dictionary(contig=chromosome_name,
                                            start_pos=0,
                                            end_pos=chromosome_lengths[chromosome_name],
                                            hom_filter=True)

            positional_variants = vcf_handler.get_variant_dictionary()
            current_chromosome_name = chromosome_name

            print("now current chromosome is:", current_chromosome_name)
            print("length:", chromosome_lengths[chromosome_name])
            sys.stdout.flush()

        # print(numpy.array2string(data[1:5,:], threshold=numpy.nan, separator="\t", precision=2, max_line_width=500))

        x = data[:,4:-1]
        coordinates = data[:,0:2]
        frequency_data = x[:,:-3]

        positional_frequency_data = get_positional_frequency_data(coordinates, frequency_data)

        n_false_negative, n_true_positive, false_negatives = \
            validate_positional_variants(positional_variants=positional_variants,
                                         positional_frequency_data=positional_frequency_data,
                                         confident_interval_tree=interval_trees_by_chromosome[chromosome_name],
                                         start=start,
                                         stop=stop)

        total_true_positive += n_true_positive
        total_false_negative += n_false_negative
        all_false_negatives.extend(false_negatives)

        if n_false_negative > 0:
            print(chromosome_name, start, stop, n_false_negative, n_true_positive)
            for false_negative in false_negatives:
                print(false_negative[:-1])
                for record in false_negative[-1]:
                    print(record)

        # --- test ---
        # numpy.savetxt('test.out', data, delimiter='\t')
        # if i > 20:
        #     break

    print("FN: ", total_false_negative)
    print("TP: ", total_true_positive)

    # for false_negative in false_negatives:
    #     print(false_negative[:-1])
    #     for record in false_negative[-1]:
    #         print(record)


parent_directory_path = "/home/ryan/data/GIAB/filter_model_training_data/vision/WG/0_threshold/confident/chr1_19__0_all_1_coverage"
vcf_path = "/home/ryan/data/GIAB/NA12878_GRCh37.vcf.gz"
fasta_path = "/home/ryan/data/GIAB/GRCh37_WG.fa"
bed_path = "/home/ryan/data/GIAB/NA12878_GRCh37_confident.bed"
file_extension = ".npz"

file_paths = FileManager.get_all_filepaths_by_type(parent_directory_path=parent_directory_path,
                                                   file_extension=file_extension)

validate_regional_candidate_data(file_paths, vcf_path, fasta_path, bed_path)
