from modules.core.VCFGraphGenerator import VCFGraphGenerator
from modules.core.AlignGraphCandidateFinder import CandidateFinder
from modules.core.AlignmentGraph import AlignmentGraph
from modules.core.AlignmentGraphLabeler import AlignmentGraphLabeler
from modules.handlers.FastaHandler import FastaHandler
from modules.handlers.VcfHandler import VCFFileProcessor
from modules.handlers.BamHandler import BamHandler
from matplotlib import pyplot
import numpy
import sys

def test_with_realtime_BAM_data():
    chromosome_name = "1"
    chromosome_name = "chr" + chromosome_name

    # --- chr3 PG ---
    # start_position = 73600    # insert
    # end_position = 73625

    # start_position = 77020      # false double alt
    # end_position = 77045

    # start_position = 77760      # long het delete
    # end_position = 77770

    # --- chr19 GRCh37 ---
    # start_position = 3039220      # long het delete
    # end_position = 3039241

    # start_position = 3039220      # not a true variant? ins + del
    # end_position = 3039224

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

    # start_position = 100000000      # arbitrary test region ... takes about 2-3 min to build graph (July 6)
    # end_position = 101000000

    # --- chr1 GRCh37 ---
    # start_position = 100774306      #   100774311	T	C	50	PASS
    # end_position = 100774315

    # start_position = 100921841      #   100921846	T	A	50	PASS  + 100921847	T	A	50	PASS
    # end_position = 100921852

    # start_position = 100885615      #   100885619	.	C	CACACATAT,CACATATAT	50	PASS
    # end_position = 100885623

    # start_position = 100717470        # 100717475	rs75444938	TTTAGTTATA	T	50	PASS
    # end_position = 100717486

    # start_position = 100285630        # 100285631	.	GGAAA	GGAAAGAAA,G	50	PASS
    # end_position = 100285635

    # start_position = 100332353        # 100332356	rs145029209	T	TTTTCTTTATTTA	50	PASS
    # end_position = 100332390

    # ---- ILLUMINA (from personal laptop) ------------------------------------
    # bam_file_path = "/Users/saureous/data/Platinum/chr3_200k.bam"
    # reference_file_path = "/Users/saureous/data/Platinum/chr3.fa"
    # vcf_path = "/Users/saureous/data/Platinum/NA12878_S1.genome.vcf.gz"

    # ---- GIAB (dev machine) -------------------------------------------------
    # bam_file_path = "/home/ryan/data/GIAB/NA12878_GIAB_30x_GRCh37.sorted.bam"
    # reference_file_path = "/home/ryan/data/GIAB/GRCh37_WG.fa"
    # vcf_path = "/home/ryan/data/GIAB/NA12878_GRCh37.vcf.gz"

    # ---- Nanopore (dev machine) ---------------------------------------------
    bam_file_path = "/home/ryan/data/Nanopore/whole_genome_nanopore.bam"
    reference_file_path = "/home/ryan/data/GIAB/GRCh38_WG.fa"
    vcf_path = "/home/ryan/data/GIAB/NA12878_GRCh38_PG.vcf.gz"
    # -------------------------------------------------------------------------

    vcf_handler = VCFFileProcessor(vcf_path)

    start_position = 160604860
    end_position = 160604880

    vcf_handler.populate_dictionary(contig=chromosome_name,
                                    start_pos=start_position,
                                    end_pos=end_position)

    positional_variants = vcf_handler.get_variant_dictionary()

    all_true_frequencies = list()
    all_true_cigar_codes = list()
    all_false_frequencies = list()
    all_false_cigar_codes = list()

    for position in positional_variants:
        # if position != 160604871:
        #     continue

        sys.stdout.write('\r'+ str(position))

        start_position = position - 1
        end_position = position + 10

        alignment_graph = generate_alignment_graph(reference_file_path=reference_file_path,
                                                   bam_file_path=bam_file_path,
                                                   vcf_path=vcf_path,
                                                   chromosome_name=chromosome_name,
                                                   start_position=start_position,
                                                   end_position=end_position)

        # get stats on nodes
        true_frequencies, true_cigar_codes, false_frequencies, false_cigar_codes = \
            alignment_graph.get_node_stats()

        all_true_frequencies.extend(true_frequencies)
        all_true_cigar_codes.extend(true_cigar_codes)
        all_false_frequencies.extend(false_frequencies)
        all_false_cigar_codes.extend(false_cigar_codes)

    # plot frequencies for labeled true and false nodes
    figure, (axes1, axes2) = pyplot.subplots(nrows=2, sharex=True)

    step = 0.02
    bins = numpy.arange(0, 1 + step, step=step)
    frequencies_true, bins1 = numpy.histogram(all_true_frequencies, bins=bins)
    frequencies_false, bins2 = numpy.histogram(all_false_frequencies, bins=bins)

    center = (bins[:-1] + bins[1:]) / 2

    axes1.bar(center, frequencies_true, width=step, align="center")
    axes2.bar(center, frequencies_false, width=step, align="center")

    # plot distribution of cigar operations for true and false nodes
    figure, (axes1, axes2) = pyplot.subplots(nrows=2, sharex=True)

    step = 1
    bins = numpy.arange(0, 4+step, step=step)
    cigar_frequencies_true, bins1 = numpy.histogram(all_true_cigar_codes, bins=bins)
    cigar_frequencies_false, bins2 = numpy.histogram(all_false_cigar_codes, bins=bins)

    center = (bins[:-1] + bins[1:]) / 2

    axes1.bar(center, cigar_frequencies_true, width=step, align="center")
    axes2.bar(center, cigar_frequencies_false, width=step, align="center")

    pyplot.show()


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

    graph_candidate_finder.get_read_alignment_data(reads=reads)

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
