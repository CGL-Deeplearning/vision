from modules.core.CandidateFinder import CandidateFinder
from modules.core.GraphCandidateFinder import CandidateFinder as GraphCandidateFinder
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


def test_with_realtime_BAM_data():
    chromosome_name = "chr3"
    # start_position = 73600    # insert
    # end_position = 73625
    # start_position = 77020      # false double alt
    # end_position = 77045
    start_position = 77760      # long het delete
    end_position = 77770

    bam_file_path = "/Users/saureous/data/Platinum/chr3_200k.bam"
    reference_file_path = "/Users/saureous/data/Platinum/chr3.fa"
    vcf_path = "/Users/saureous/data/Platinum/NA12878_S1.genome.vcf.gz"

    bam_handler = BamHandler(bam_file_path)
    fasta_handler = FastaHandler(reference_file_path)
    vcf_handler = VCFFileProcessor(file_path=vcf_path)
    alignment_graph = AlignmentGraph(chromosome_name=chromosome_name,
                                     start_position=start_position,
                                     end_position=end_position)

    # get the reads that fall in that region
    reads = bam_handler.get_reads(chromosome_name=chromosome_name,
                                  start=start_position,
                                  stop=end_position)

    # create candidate finder object
    candidate_finder = GraphCandidateFinder(reads=reads,
                                            fasta_handler=fasta_handler,
                                            chromosome_name=chromosome_name,
                                            region_start_position=start_position,
                                            region_end_position=end_position,
                                            alignment_graph=alignment_graph)

    candidate_finder.get_read_alignment_data(reads=reads)
    alignment_graph.print_alignment_graph()
    alignment_graph.plot_alignment_graph()


if __name__ == "__main__":
    # test_with_artifical_data()
    # test_with_positional_BAM_data()
    test_with_realtime_BAM_data()
