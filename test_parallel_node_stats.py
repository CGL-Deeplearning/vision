from modules.core.AlignGraphCandidateFinder import CandidateFinder
from modules.core.AlignmentGraph import AlignmentGraph
from modules.core.AlignmentGraphLabeler import AlignmentGraphLabeler
from modules.core.IntervalTree import IntervalTree
from modules.core.IterativeHistogram import IterativeHistogram
from modules.handlers.FastaHandler import FastaHandler
from modules.handlers.VcfHandler import VCFFileProcessor
from modules.handlers.BamHandler import BamHandler
from modules.handlers.TsvHandler import TsvHandler
from matplotlib import pyplot
from multiprocessing import Pool
import multiprocessing
import numpy
import sys
import os

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


def parallel_run(reference_file_path, bed_file_path, vcf_file_path, bam_file_path, chromosome_name, start_position, end_position, return_dict, counter, n_chunks):
    chromosomal_interval_tree = build_chromosomal_interval_trees(bed_file_path)

    vcf_handler = VCFFileProcessor(vcf_file_path)

    vcf_handler.populate_dictionary(contig=chromosome_name,
                                    start_pos=start_position,
                                    end_pos=end_position)

    positional_variants = vcf_handler.get_variant_dictionary()

    histogram = IterativeHistogram(start=0, stop=60, n_bins=12, unbounded_upper_bin=True, unbounded_lower_bin=True)
    shape = histogram.get_histogram().shape

    # print(shape)

    all_cigar_codes = [[],[]]
    all_frequencies = [[[],[]] for code in [REF, SNP, INS, DEL]]
    all_qualities = [[numpy.zeros(shape), numpy.zeros(shape)] for code in [REF, SNP, INS, DEL]]

    for p,position in enumerate(positional_variants):
        interval = [position,position]
        if interval not in chromosomal_interval_tree[chromosome_name]:
            continue

        # sys.stdout.write('\r' + str(p/len(positional_variants)))

        start_position = position - 2
        end_position = position + 12

        alignment_graph = generate_alignment_graph(reference_file_path=reference_file_path,
                                                   bam_file_path=bam_file_path,
                                                   vcf_path=vcf_file_path,
                                                   chromosome_name=chromosome_name,
                                                   start_position=start_position,
                                                   end_position=end_position)

        if alignment_graph is None:
            return

        # get stats on nodes
        cigar_codes, frequencies, quality_histograms = alignment_graph.get_node_stats()

        for label in [0, 1]:
            all_cigar_codes[label].extend(cigar_codes[label])

            for cigar_code in [REF, SNP, INS, DEL]:
                all_frequencies[cigar_code][label].extend(frequencies[cigar_code][label])

                # print(all_qualities[cigar_code][label])
                # print(quality_histograms[cigar_code][label])
                all_qualities[cigar_code][label] += quality_histograms[cigar_code][label]

                # print(cigar_code, frequencies[cigar_code])

    counter.value += 1

    sys.stdout.write('\r' + "%.2f%% Completed"%(100*counter.value/n_chunks))

    return_dict[start_position] = [all_cigar_codes, all_frequencies, all_qualities]

    return


def test_with_realtime_BAM_data():
    chromosome_name = "18"

    # --- chr1 GRCh37 ---
    # start_position = 93839128 - 2     # no >0 MQ reads??
    # end_position = 93839128 + 10

    # --- chr1 GRCh38 ---
    # start_position = 100816140      # chr1 100816142 . TG T 50 PASS
    # end_position = 100816145

    # start_position = 100822960      # chr1 100822965 . A T 50 PASS
    # end_position = 100822969

    # start_position = 101114275      # chr1 101114279 . C T 50 PASS
    # end_position = 101114280

    # start_position = 100866200      # chr1 100866204 . G A 50 PASS
    # end_position = 100866209

    # start_position = 100915190        # 100915193	.	TTTTC	TTTTCTTTCTTTC,T	50	PASS
    # end_position = 100915200

    # start_position = 54091130           # staggered overlapping long delete and het SNP
    # end_position = 54091180

    # start_position = 100000000      # arbitrary test region ... takes about 2-3 min to build graph (July 6)
    # end_position = 101000000

    # ---- ILLUMINA (from personal laptop) ------------------------------------
    # bam_file_path = "/Users/saureous/data/Platinum/chr3_200k.bam"
    # reference_file_path = "/Users/saureous/data/Platinum/chr3.fa"
    # vcf_file_path = "/Users/saureous/data/Platinum/NA12878_S1.genome.vcf.gz"

    # ---- GIAB (dev machine) -------------------------------------------------
    # bam_file_path = "/home/ryan/data/GIAB/NA12878_GIAB_30x_GRCh37.sorted.bam"
    # reference_file_path = "/home/ryan/data/GIAB/GRCh37_WG.fa"
    # vcf_file_path = "/home/ryan/data/GIAB/NA12878_GRCh37.vcf.gz"
    # bed_file_path = "/home/ryan/data/GIAB/NA12878_GRCh37_confident_chr18.bed"

    # ---- Nanopore GUPPY (dev machine) ---------------------------------------
    bam_file_path = "/home/ryan/data/Nanopore/BAM/Guppie/rel5-guppy-0.3.0-chunk10k.sorted.bam"
    reference_file_path = "/home/ryan/data/GIAB/GRCh38_WG.fa"
    vcf_file_path = "/home/ryan/data/GIAB/NA12878_GRCh38_PG.vcf.gz"
    bed_file_path = "/home/ryan/data/GIAB/NA12878_GRCh38_confident_chr18.bed"

    chromosome_name = "chr" + chromosome_name
    # -------------------------------------------------------------------------

    if sys.platform == 'win32':
        available_threads = int(os.environ['NUMBER_OF_PROCESSORS'])
    else:
        available_threads = int(os.popen('grep -c cores /proc/cpuinfo').read())

    max_threads = available_threads - 4
    # max_threads = 1

    start_position = 6000000
    end_position = 8400000
    chunk_size = 2500000

    steps = range(start_position, end_position + chunk_size, chunk_size)
    n_chunks = len(steps)

    # print(list(steps))

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

    histogram = IterativeHistogram(start=0, stop=60, n_bins=12, unbounded_upper_bin=True, unbounded_lower_bin=True)
    edges = histogram.edges
    shape = histogram.get_histogram().shape

    # recombine dictionary of thread-specific data to a universal dictionary
    all_cigar_codes = [[],[]]
    all_frequencies = [[[],[]] for code in [REF, SNP, INS, DEL]]
    all_qualities = [[numpy.zeros(shape), numpy.zeros(shape)] for code in [REF, SNP, INS, DEL]]

    for key in return_dict.keys():
        cigar_codes, frequencies, qualities = return_dict[key]

        for label in [0, 1]:
            all_cigar_codes[label].extend(cigar_codes[label])

            for cigar_code in [REF, SNP, INS, DEL]:
                all_frequencies[cigar_code][label].extend(frequencies[cigar_code][label])
                all_qualities[cigar_code][label] += qualities[cigar_code][label]

    # plot frequencies for labeled true and false nodes
    for cigar_code in [REF, SNP, INS, DEL]:
        figure, (axes1, axes2) = pyplot.subplots(nrows=2, sharex=True)

        all_false_frequencies, all_true_frequencies = all_frequencies[cigar_code]

        if cigar_code == INS:
            print(all_false_frequencies)

        step = 0.02
        bins = numpy.arange(0, 1 + step, step=step)
        frequencies_true, bins1 = numpy.histogram(all_true_frequencies, bins=bins)
        frequencies_false, bins2 = numpy.histogram(all_false_frequencies, bins=bins)

        center = (bins[:-1] + bins[1:]) / 2

        axes1.bar(center, frequencies_true, width=step, align="center")
        axes2.bar(center, frequencies_false, width=step, align="center")

        axes1.set_ylabel("Count (True)")
        axes2.set_ylabel("Count (False)")
        axes2.set_xlabel("Node Frequency")
        pyplot.savefig("frequencies_" + cigar_code_names[cigar_code] + ".png")

    # plot qualities for labeled true and false nodes
    for cigar_code in [REF, SNP, INS, DEL]:
        figure, (axes1, axes2) = pyplot.subplots(nrows=2, sharex=True)

        all_false_qualities, all_true_qualities = all_qualities[cigar_code]

        step = edges[1] - edges[0]
        bins = numpy.array(edges)
        center = (bins[:-1] + bins[1:]) / 2

        axes1.bar(center, all_true_qualities, width=step, align="center")
        axes2.bar(center, all_false_qualities, width=step, align="center")

        axes1.set_ylabel("Count (True)")
        axes2.set_ylabel("Count (False)")
        axes2.set_xlabel("Base Quality")
        pyplot.savefig("qualities_" + cigar_code_names[cigar_code] + ".png")

    # plot distribution of cigar operations for true and false nodes
    figure, (axes1, axes2) = pyplot.subplots(nrows=2, sharex=True)

    all_false_cigar_codes, all_true_cigar_codes = all_cigar_codes

    step = 1
    bins = numpy.arange(0, 4+step, step=step)
    cigar_frequencies_true, bins1 = numpy.histogram(all_true_cigar_codes, bins=bins)
    cigar_frequencies_false, bins2 = numpy.histogram(all_false_cigar_codes, bins=bins)

    center = (bins[:-1] + bins[1:]) / 2

    axes1.bar(center, cigar_frequencies_true, width=step, align="center")
    axes2.bar(center, cigar_frequencies_false, width=step, align="center")

    axes2.set_xticks([0.5, 1.5, 2.5, 3.5])
    axes2.set_xticklabels(["REF","SNP","INS","DEL"])
    axes2.set_xlabel("Cigar Type")
    axes1.set_ylabel("Count (True)")
    axes2.set_ylabel("Count (False)")

    pyplot.savefig("cigars.png")

    return


def generate_alignment_graph(reference_file_path, vcf_path, bam_file_path, chromosome_name, start_position, end_position):
    fasta_handler = FastaHandler(reference_file_path)
    bam_handler = BamHandler(bam_file_path)
    fasta_handler = FastaHandler(reference_file_path)
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

    return alignment_graph


if __name__ == "__main__":
    test_with_realtime_BAM_data()
