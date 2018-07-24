from modules.core.VCFGraphGenerator import VCFGraphGenerator
from modules.core.AlignGraphCandidateFinder import CandidateFinder
from modules.core.AlignmentGraph import AlignmentGraph
from modules.core.AlignmentGraphLabeler import AlignmentGraphLabeler
from modules.handlers.FastaHandler import FastaHandler
from modules.handlers.VcfHandler import VCFFileProcessor
from modules.handlers.BamHandler import BamHandler
from matplotlib import pyplot
import numpy


def test_with_realtime_BAM_data():
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

    # start_position = 54091130           # staggered overlapping long delete and het SNP
    # end_position = 54091180

    # start_position = 100000000      # arbitrary test region ... takes about 2-3 min to build graph (July 6)
    # end_position = 101000000

    start_position = 261528 - 1
    end_position = 261528 + 3

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

    chromosome_name = "19"

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
    chromosome_name = "chr" + chromosome_name
    # -------------------------------------------------------------------------

    vcf_handler = VCFFileProcessor(vcf_path)

    # collecting 1 extra vcf entry doesn't cause conflicts, because query keys are positional
    vcf_handler.populate_dictionary(contig=chromosome_name,
                                    start_pos=start_position,
                                    end_pos=end_position)

    positional_variants = vcf_handler.get_variant_dictionary()

    for position in positional_variants:
        print(position)

        start_position = position - 3
        end_position = position + 10

        figure = pyplot.figure()
        axes = pyplot.axes()

        vcf_graph = generate_vcf_graph(reference_file_path=reference_file_path,
                                       vcf_path=vcf_path,
                                       chromosome_name=chromosome_name,
                                       start_position=start_position,
                                       end_position=end_position)

        axes, x_limits, y_limits = visualize_graph(alignment_graph=vcf_graph, axes=axes)

        axes.set_aspect("equal")

        y_lower = y_limits[0]
        y_upper = y_limits[1]

        x_lower = x_limits[0]
        x_upper = x_limits[1]

        axes.set_xlim(x_lower, x_upper)
        axes.set_ylim(y_lower, y_upper)

        pyplot.show()


def visualize_graph(alignment_graph, axes):
    axes, x_limits, y_limits = alignment_graph.plot_alignment_graph(axes=axes, show=False, set_axis_limits=False)

    pileup_string = alignment_graph.generate_pileup()
    print("\nBAM GRAPH:")
    print(pileup_string)

    matrix = alignment_graph.generate_adjacency_matrix()
    matrix_label = alignment_graph.generate_adjacency_matrix(label_variants=True)

    numpy.savetxt("test_matrix.tsv", matrix, fmt="%d", delimiter='\t')
    numpy.savetxt("test_matrix_label.tsv", matrix_label, fmt="%d", delimiter='\t')

    return axes, x_limits, y_limits


def generate_vcf_graph(reference_file_path, vcf_path, chromosome_name, start_position, end_position):
    fasta_handler = FastaHandler(reference_file_path)
    vcf_handler = VCFFileProcessor(vcf_path)

    # candidate finder includes end position, so should the reference sequence
    reference_sequence = fasta_handler.get_sequence(chromosome_name=chromosome_name,
                                                    start=start_position,
                                                    stop=end_position+1)

    # collecting 1 extra vcf entry doesn't cause conflicts, because query keys are positional
    vcf_handler.populate_dictionary(contig=chromosome_name,
                                    start_pos=start_position,
                                    end_pos=end_position+1)

    positional_variants = vcf_handler.get_variant_dictionary()

    alignment_graph = AlignmentGraph(chromosome_name=chromosome_name,
                                     start_position=start_position,
                                     end_position=end_position)

    # create graphbuilder object which iterates and parses the reference+VCF
    vcf_graph_builder = VCFGraphGenerator(reference_sequence=reference_sequence,
                                          positional_variants=positional_variants,
                                          chromosome_name=chromosome_name,
                                          start_position=start_position,
                                          end_position=end_position,
                                          graph=alignment_graph)

    vcf_graph_builder.parse_region()

    # vcf_alignment_graph.print_alignment_graph()
    #
    # pileup_string = vcf_alignment_graph.generate_pileup()
    #
    # print(pileup_string)

    pileup_string = alignment_graph.generate_pileup()
    print("\nVCF GRAPH:")
    print(pileup_string)

    # alignment_graph.generate_adjacency_matrix()
    # alignment_graph.generate_adjacency_matrix(label_variants=True)

    return alignment_graph


def generate_alignment_graph(reference_file_path, vcf_path, bam_file_path, chromosome_name, start_position, end_position, axes):
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
