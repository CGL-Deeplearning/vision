from modules.core.VCFGraphGenerator import VCFGraphGenerator
from modules.core.AlignmentGraph import AlignmentGraph
from modules.handlers.FastaHandler import FastaHandler
from modules.handlers.VcfHandler import VCFFileProcessor
import random


def test_with_realtime_BAM_data():
    chromosome_name = "1"
    # chromosome_name = "chr" + chromosome_name

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

    # start_position = 100000000      # arbitrary test region ... takes about 2-3 min to build graph (July 6)
    # end_position = 101000000

    # --- chr1 GRCh37 ---
    # start_position = 100774306      #   100774311	T	C	50	PASS
    # end_position = 100774315

    # start_position = 100921841      #   100921846	T	A	50	PASS  + 100921847	T	A	50	PASS
    # end_position = 100921852

    # start_position = 100885615      #   100885619	.	C	CACACATAT,CACATATAT	50	PASS
    # end_position = 100885623

    start_position = 100717470  # 100717475	rs75444938	TTTAGTTATA	T	50	PASS
    end_position = 100717479

    # ---- ILLUMINA (from personal laptop) ------------------------------------
    # bam_file_path = "/Users/saureous/data/Platinum/chr3_200k.bam"
    # reference_file_path = "/Users/saureous/data/Platinum/chr3.fa"
    # vcf_path = "/Users/saureous/data/Platinum/NA12878_S1.genome.vcf.gz"

    # ---- GIAB (dev machine) -------------------------------------------------
    bam_file_path = "/home/ryan/data/GIAB/NA12878_GIAB_30x_GRCh37.sorted.bam"
    reference_file_path = "/home/ryan/data/GIAB/GRCh37_WG.fa"
    vcf_path = "/home/ryan/data/GIAB/NA12878_GRCh37.vcf.gz"

    # ---- Nanopore (dev machine) ---------------------------------------------
    # bam_file_path = "/home/ryan/data/Nanopore/whole_genome_nanopore.bam"
    # reference_file_path = "/home/ryan/data/GIAB/GRCh38_WG.fa"
    # vcf_path = "/home/ryan/data/GIAB/NA12878_GRCh38_PG.vcf.gz"
    # -------------------------------------------------------------------------

    fasta_handler = FastaHandler(reference_file_path)
    vcf_handler = VCFFileProcessor(file_path=vcf_path)

    # candidate finder includes end position, so should the reference sequence
    reference_sequence = fasta_handler.get_sequence(chromosome_name=chromosome_name,
                                                    start=start_position,
                                                    stop=end_position+1)

    # collecting 1 extra vcf entry doesn't cause conflicts, because queries are positional
    vcf_handler.populate_dictionary(contig=chromosome_name,
                                    start_pos=start_position,
                                    end_pos=end_position+1)

    positional_variants = vcf_handler.get_variant_dictionary()

    alignment_graph = AlignmentGraph(chromosome_name=chromosome_name,
                                     start_position=start_position,
                                     end_position=end_position)

    # create graphbuilder object which iterates and parses the reference+VCF
    graph_builder = VCFGraphGenerator(reference_sequence=reference_sequence,
                                      positional_variants=positional_variants,
                                      chromosome_name=chromosome_name,
                                      start_position=start_position,
                                      end_position=end_position,
                                      graph=alignment_graph)

    graph_builder.parse_region()
    alignment_graph.print_alignment_graph()
    alignment_graph.plot_alignment_graph()

if __name__ == "__main__":
    test_with_realtime_BAM_data()
