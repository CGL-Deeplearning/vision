from modules.core.KmerDataCollector import KmerDataCollector
from modules.core.KmerGraph import KmerGraph
from modules.handlers.BamHandler import BamHandler
from modules.handlers.FastaHandler import FastaHandler
from modules.handlers.VcfHandler import VCFFileProcessor
import numpy
from matplotlib import pyplot
from matplotlib.pyplot import cm
import random


numpy.set_printoptions(suppress=True)


def plot_kmer_distribution(kmer_frequencies, axes, color, label):
    kmer_frequencies = list(kmer_frequencies.values())
    y_max = 2000
    y_min = 0

    bins = numpy.arange(y_min, y_max, step=1)
    frequencies, bins = numpy.histogram(kmer_frequencies, bins=bins)

    frequencies = numpy.log10(frequencies)
    centers = (bins[:-1] + bins[1:])/2

    print(centers.shape, frequencies.shape)

    print("y")
    print(numpy.array2string(frequencies[:30], precision=2, separator='\t'))

    axes.plot(centers, frequencies, lw=1.2, color=color, label=label)

    return axes


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

    # --- chr19 ---
    # start_position = 3039220      # long het delete
    # end_position = 3039241

    # --- chr1 ---
    # start_position = 100816140      # chr1 100816142 . TG T 50 PASS
    # end_position = 100816145

    # start_position = 100822960      # chr1 100822965 . A T 50 PASS
    # end_position = 100822969

    # start_position = 101114275      # chr1 101114279 . C T 50 PASS
    # end_position = 101114285

    start_position = 1000000
    end_position = 1000000

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

    k_start = 19
    k_end = 23
    k_step_size = 2
    k_range = k_end - k_start
    k_steps = range(k_start, k_end+k_step_size, 2)
    color_range = cm.viridis(numpy.linspace(0, 1, int(round(k_range/k_step_size))+2))

    # print(k_range)
    # print(list(k_steps))
    # print(k_range/k_step_size)
    # print(color_range.shape)
    # print(color_range)

    axes = pyplot.axes()
    for i,k in enumerate(k_steps):
        bam_handler = BamHandler(bam_file_path)
        fasta_handler = FastaHandler(reference_file_path)
        vcf_handler = VCFFileProcessor(file_path=vcf_path)

        # get the reads that fall in that region
        reads = bam_handler.get_reads(chromosome_name=chromosome_name,
                                      start=start_position,
                                      stop=end_position)

        kmer_graph = KmerGraph(chromosome_name=chromosome_name,
                               start_position=start_position,
                               end_position=end_position,
                               k=k)

        # create candidate finder object
        kmer_data_collector = KmerDataCollector(reads=reads,
                                                fasta_handler=fasta_handler,
                                                chromosome_name=chromosome_name,
                                                region_start_position=start_position,
                                                region_end_position=end_position,
                                                kmer_graph=kmer_graph,
                                                k=k)

        kmer_frequencies = kmer_data_collector.get_read_data(reads=reads)

        print(k, len(kmer_frequencies))

        color = color_range[i,:]
        axes = plot_kmer_distribution(kmer_frequencies=kmer_frequencies, axes=axes, color=color, label="k = "+str(k))

    axes.legend()
    pyplot.show()


if __name__ == "__main__":
    # test_with_artifical_data()
    # test_with_positional_BAM_data()
    test_with_realtime_BAM_data()
