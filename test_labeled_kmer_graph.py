from modules.core.VCFGraphGenerator import VCFGraphGenerator
from modules.core.VCFGraphGenerator import READ_IDS
from modules.core.KmerGraphCandidateFinder import CandidateFinder
from modules.core.KmerGraphLabeler import KmerGraphLabeler
from modules.core.KmerGraph import KmerGraph
from modules.core.AlignmentGraph import AlignmentGraph
from modules.handlers.FastaHandler import FastaHandler
from modules.handlers.VcfHandler import VCFFileProcessor
from modules.handlers.BamHandler import BamHandler
from modules.handlers.FileManager import FileManager
from matplotlib import pyplot
from os import path


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

    chromosome_name = "19"

    # ---- ILLUMINA (from personal laptop) ------------------------------------
    # bam_file_path = "/Users/saureous/data/Platinum/chr3_200k.bam"
    # reference_file_path = "/Users/saureous/data/Platinum/chr3.fa"
    # vcf_path = "/Users/saureous/data/Platinum/NA12878_S1.genome.vcf.gz"

    # ---- GIAB (dev machine) -------------------------------------------------
    # bam_file_path = "/home/ryan/data/GIAB/NA12878_GIAB_30x_GRCh37.sorted.bam"
    # reference_file_path = "/home/ryan/data/GIAB/GRCh37_WG.fa"
    # vcf_path = "/home/ryan/data/GIAB/NA12878_GRCh37.vcf.gz"

    # ---- Nanopore ALBACORE (dev machine) ---------------------------------------------
    bam_file_path = "/home/ryan/data/Nanopore/BAM/Albacore/whole_genome_nanopore.bam"
    reference_file_path = "/home/ryan/data/GIAB/GRCh38_WG.fa"
    vcf_path = "/home/ryan/data/Nanopore/BAM_VCF_concordance/chr19_FN_only.vcf.gz"

    chromosome_name = "chr" + chromosome_name

    # ---- Nanopore GUPPY (dev machine) ---------------------------------------------
    # bam_file_path = "/home/ryan/data/Nanopore/BAM/Guppie/rel5-guppy-0.3.0-chunk10k.sorted.bam"
    # reference_file_path = "/home/ryan/data/GIAB/GRCh38_WG.fa"
    # vcf_path = "/home/ryan/data/Nanopore/BAM_VCF_concordance/chr19_FN_only.vcf.gz"
    #
    # chromosome_name = "chr" + chromosome_name
    # -------------------------------------------------------------------------

    output_directory = "output/labeled_graphs/"
    FileManager.ensure_directory_exists(output_directory)

    vcf_handler = VCFFileProcessor(vcf_path)

    start_position = 267200
    end_position = 58584910

    # collecting 1 extra vcf entry doesn't cause conflicts, because query keys are positional
    vcf_handler.populate_dictionary(contig=chromosome_name,
                                    start_pos=start_position,
                                    end_pos=end_position)

    positional_variants = vcf_handler.get_variant_dictionary()

    print(start_position, end_position)
    print(len(positional_variants))

    k=7

    for position in positional_variants:
        # if position != 261528:
        #     continue

        start_position = position - k - 1
        end_position = position + 10

        figure, (axes1, axes2) = pyplot.subplots(nrows=2, sharex=True, sharey=True)

        vcf_graph = generate_vcf_graph(reference_file_path=reference_file_path,
                                       vcf_path=vcf_path,
                                       chromosome_name=chromosome_name,
                                       start_position=start_position,
                                       end_position=end_position)

        haplotypes = get_haplotypes(vcf_graph)

        kmer_graph = generate_kmer_graph(reference_file_path=reference_file_path,
                                         bam_file_path=bam_file_path,
                                         vcf_path=vcf_path,
                                         chromosome_name=chromosome_name,
                                         start_position=start_position,
                                         end_position=end_position,
                                         k=k,
                                         haplotypes=haplotypes)

        save_visual_graph(alignment_graph=kmer_graph,
                          vcf_graph=vcf_graph,
                          figure=figure,
                          axes1=axes1,
                          axes2=axes2,
                          chromosome_name=chromosome_name,
                          position=position,
                          output_directory=output_directory)

        # save_pileup(alignment_graph=kmer_graph,
        #             vcf_graph=vcf_graph,
        #             chromosome_name=chromosome_name,
        #             position=position,
        #             output_directory=output_directory)

        pyplot.close()

        print(position)


def get_haplotypes(vcf_graph):
    # extract haplotype sequences from vcf_graph pileup
    read_sequences = vcf_graph.get_read_sequences()

    haplotypes = read_sequences.values()

    # print(read_sequences)
    # sequence1 = read_sequences[READ_IDS[0]]
    # sequence2 = read_sequences[READ_IDS[1]]
    # print(sequence1)
    # print(sequence2)

    return haplotypes


def save_visual_graph(alignment_graph, vcf_graph, figure, axes1, axes2, chromosome_name, position, output_directory):
    axes1, x_limits1, y_limits1 = visualize_graph(alignment_graph=alignment_graph, axes=axes1)
    # axes2, x_limits3, y_limits3 = visualize_graph(alignment_graph=vcf_graph, axes=axes2)

    axes2.set_aspect("equal")
    axes1.set_aspect("equal")

    # y_lower = min(y_limits1[0], y_limits3[0])
    # y_upper = max(y_limits1[1], y_limits3[1])
    #
    # x_lower = min(x_limits1[0], x_limits3[0])
    # x_upper = max(x_limits1[1], x_limits3[1])

    # axes1.set_xlim(x_lower, x_upper)
    # axes2.set_xlim(x_lower, x_upper)
    #
    # axes1.set_ylim(y_lower, y_upper)
    # axes2.set_ylim(y_lower, y_upper)

    figure.set_size_inches(12,8)

    figure_name = chromosome_name + "_" + str(position)

    axes1.set_title(figure_name)

    output_path = path.join(output_directory, figure_name + ".png")
    pyplot.savefig(output_path)

    pyplot.show()


def save_pileup(alignment_graph, vcf_graph, chromosome_name, position, output_directory):
    bam_pileup_string = alignment_graph.generate_pileup()
    vcf_pileup_string = vcf_graph.generate_pileup()

    output_string = '\n'.join(["BAM GRAPH:", bam_pileup_string, "\nVCF GRAPH:", vcf_pileup_string])

    pileup_name = chromosome_name + "_" + str(position)
    output_path = path.join(output_directory, pileup_name + ".txt")

    with open(output_path, 'w') as file:
        file.write(output_string)


def visualize_graph(alignment_graph, axes):
    axes, x_limits, y_limits = alignment_graph.plot_graph(axes=axes, show=False, set_axis_limits=True)

    # matrix = alignment_graph.generate_adjacency_matrix()
    # matrix_label = alignment_graph.generate_adjacency_matrix(label_variants=True)
    # numpy.savetxt("test_matrix.tsv", matrix, fmt="%d", delimiter='\t')
    # numpy.savetxt("test_matrix_label.tsv", matrix_label, fmt="%d", delimiter='\t')

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

    return alignment_graph


def generate_kmer_graph(reference_file_path, vcf_path, bam_file_path, chromosome_name, start_position, end_position, k, haplotypes):
    fasta_handler = FastaHandler(reference_file_path)
    bam_handler = BamHandler(bam_file_path)
    fasta_handler = FastaHandler(reference_file_path)
    vcf_handler = VCFFileProcessor(vcf_path)

    reference_sequence = fasta_handler.get_sequence(chromosome_name=chromosome_name,
                                                    start=start_position,
                                                    stop=end_position+1)

    kmer_graph = KmerGraph(chromosome_name=chromosome_name,
                           start_position=start_position,
                           end_position=end_position,
                           k=k)

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
                                             kmer_graph=kmer_graph,
                                             k=k)

    graph_candidate_finder.get_read_alignment_data(reads=reads)

    labeler = KmerGraphLabeler(kmer_graph=kmer_graph, haplotypes=haplotypes)

    labeler.parse_region()

    return kmer_graph


if __name__ == "__main__":
    test_with_realtime_BAM_data()
