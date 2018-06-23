from modules.core.CandidateFinder import CandidateFinder
from modules.core.AlignmentGraph import AlignmentGraph
from modules.handlers.BamHandler import BamHandler
from modules.handlers.FastaHandler import FastaHandler
from modules.handlers.VcfHandler import VCFFileProcessor
import random


def test_with_artifical_data():
    characters = ["A", "C", "G", "T"]

    sequence_length = 8
    sequence = [random.choice(characters) for i in range(sequence_length)]
    positional_reference = {pos: sequence for pos, sequence in enumerate(sequence)}

    print(positional_reference)

    graph = AlignmentGraph(chromosome_name="chr1",
                           start_position=0,
                           end_position=sequence_length - 1,
                           positional_reference=positional_reference)

    graph.initialize_graph_with_reference()
    graph.print_alignment_graph()


def test_with_BAM():
    chromosome_name = "19"
    start_position = 3039221
    end_position = 3039223

    bam_file_path = "/home/ryan/data/GIAB/NA12878_GIAB_30x_GRCh37.sorted.bam"
    reference_file_path = "/home/ryan/data/GIAB/GRCh37_WG.fa"
    vcf_path = "/home/ryan/data/GIAB/NA12878_GRCh37.vcf.gz"

    bam_handler = BamHandler(bam_file_path)
    fasta_handler = FastaHandler(reference_file_path)
    vcf_handler = VCFFileProcessor(file_path=vcf_path)

    # get the reads that fall in that region
    reads = bam_handler.get_reads(chromosome_name=chromosome_name,
                                  start=start_position,
                                  stop=end_position)

    # create candidate finder object
    candidate_finder = CandidateFinder(reads=reads,
                                       fasta_handler=fasta_handler,
                                       chromosome_name=chromosome_name,
                                       region_start_position=start_position,
                                       region_end_position=end_position)

    # go through each read and find positional data
    chromosome_name, \
    start_position, \
    end_position, \
    positional_reference, \
    positional_alleles, \
    positional_coverage = candidate_finder.get_positional_data(reads=reads)

    print(positional_alleles)

    graph = AlignmentGraph(chromosome_name="chr1",
                           start_position=start_position,
                           end_position=end_position,
                           positional_reference=positional_reference,
                           positional_coverage=positional_coverage,
                           positional_alleles=positional_alleles)

    graph.generate_graph_from_positional_data()
    graph.print_alignment_graph()
    graph.plot_alignment_graph()


if __name__ == "__main__":
    test_with_artifical_data()
    test_with_BAM()
