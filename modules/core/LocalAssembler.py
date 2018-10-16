from collections import defaultdict
from modules.core.ActiveRegionFinder import ActiveRegionFinder
from modules.core.DeBruijnGraph import DeBruijnGraphCreator
from modules.handlers.Read import Read
from modules.core.SSWAligner import SSWAligner
from modules.core.OptionValues import AlingerOptions, DeBruijnGraphOptions, ActiveRegionOptions, CandidateFinderOptions
"""doing this: https://software.broadinstitute.org/gatk/documentation/article.php?id=11077
A version is implemented here: https://github.com/google/deepvariant/blob/r0.7/deepvariant/realigner/aligner.py
"""


class RegionBasedHaplotypes:
    def __init__(self, haplotypes, region_start, region_end):
        self.haplotypes = haplotypes
        self.region_start = region_start
        self.region_end = region_end
        self.min_read_start = None
        self.max_read_end = None
        self.reads = []

    def assign_read(self, read):
        if self.min_read_start is None or self.max_read_end is None:
            self.min_read_start = read.reference_start
            self.max_read_end = read.reference_end

        self.min_read_start = min(self.min_read_start, read.reference_start)
        self.max_read_end = max(self.max_read_end, read.reference_end)
        self.reads.append(read)

    @staticmethod
    def overlap_length_between_ranges(range_a, range_b):
        return max(0, (min(range_a[1], range_b[1]) - max(range_a[0], range_a[0])))


class LocalAssembler:

    def __init__(self, fasta_handler, bam_handler, chromosome_name, region_start, region_end):
        self.candidate_position_weighted_sum = defaultdict(int)
        self.candidate_positions = defaultdict(int)

        self.bam_handler = bam_handler
        self.fasta_handler = fasta_handler
        self.chromosome_name = chromosome_name
        self.region_start_position = region_start
        self.region_end_position = region_end

    def perform_local_alignment(self, region_with_reads):
        if not region_with_reads.reads:
            return []
        ref_start = min(region_with_reads.min_read_start, region_with_reads.region_start) - AlingerOptions.ALIGNMENT_SAFE_BASES
        ref_end = max(region_with_reads.max_read_end, region_with_reads.region_end) + AlingerOptions.ALIGNMENT_SAFE_BASES

        if ref_end <= region_with_reads.region_end:
            return region_with_reads.reads
        else:
            ref_suffix = self.fasta_handler.get_sequence(chromosome_name=self.chromosome_name,
                                                         start=region_with_reads.region_end,
                                                         stop=ref_end)

        ref_prefix = self.fasta_handler.get_sequence(chromosome_name=self.chromosome_name,
                                                     start=ref_start,
                                                     stop=region_with_reads.region_start)
        ref = self.fasta_handler.get_sequence(chromosome_name=self.chromosome_name,
                                              start=region_with_reads.region_start,
                                              stop=region_with_reads.region_end)
        ref_seq = ref_prefix + ref + ref_suffix
        haplotypes = [ref_prefix + hap + ref_suffix for hap in region_with_reads.haplotypes]
        aligner = SSWAligner(ref_start, ref_end, ref_seq)

        realigned_reads = aligner.align_reads(haplotypes, region_with_reads.reads)

        return realigned_reads

    def perform_local_assembly(self, perform_alignment=True):
        # get the reads that fall in that region
        all_reads = self.bam_handler.get_reads(chromosome_name=self.chromosome_name,
                                               start=self.region_start_position,
                                               stop=self.region_end_position)
        # we realign all these reads
        reads_in_region = []
        for read in all_reads:
            if read.mapping_quality >= CandidateFinderOptions.MIN_MAP_QUALITY and read.is_secondary is False and \
                    read.is_supplementary is False and read.is_unmapped is False and read.is_qcfail is False:
                reads_in_region.append(Read(read))

        if perform_alignment is False:
            return reads_in_region

        # find active regions
        active_region_finder = ActiveRegionFinder(fasta_handler=self.fasta_handler,
                                                  chromosome_name=self.chromosome_name,
                                                  region_start=self.region_start_position,
                                                  region_end=self.region_end_position
                                                  )
        # mapq threshold is checked in this
        active_regions = active_region_finder.select_active_region(reads_in_region)

        assembly_active_regions = []
        possible_regions = []
        for active_region in active_regions:
            start_pos, end_pos = active_region

            if end_pos - start_pos > ActiveRegionOptions.MAX_ACTIVE_REGION_SIZE:
                continue

            graph = DeBruijnGraphCreator(start_pos, end_pos)
            ref_sequence = self.fasta_handler.get_sequence(chromosome_name=self.chromosome_name,
                                                           start=start_pos,
                                                           stop=end_pos)
            bounds = (DeBruijnGraphOptions.MIN_K, DeBruijnGraphOptions.MAX_K, DeBruijnGraphOptions.STEP_K)
            min_k, max_k = graph.find_min_k_from_ref(ref_sequence, bounds)

            if min_k is None:
                continue
            reads = self.bam_handler.get_reads(chromosome_name=self.chromosome_name,
                                               start=start_pos,
                                               stop=end_pos)
            # these reads are only for constructing the graph
            filtered_reads = []
            for read in reads:
                if read.mapping_quality >= DeBruijnGraphOptions.MIN_MAP_QUALITY and read.is_secondary is False \
                        and read.is_supplementary is False and read.is_unmapped is False and read.is_qcfail is False:
                    filtered_reads.append(Read(read))

            bounds = (min_k, max_k, DeBruijnGraphOptions.STEP_K)
            haplotypes = graph.find_haplotypes_through_linear_search_over_kmer(ref_sequence, filtered_reads, bounds)

            if not haplotypes:
                haplotypes = [ref_sequence]
            if haplotypes != [ref_sequence]:
                assembly_active_regions.append(RegionBasedHaplotypes(haplotypes, start_pos, end_pos))
                possible_regions.append((start_pos, end_pos))

        if not possible_regions:
            return reads_in_region

        # now we have the list that we filtered at the beginning of this script
        realigned_reads = list()
        for read in reads_in_region:
            read_start = read.reference_start
            read_end = ActiveRegionFinder.get_read_stop_position(read)
            read_range = (read_start, read_end)
            overlapping_lengths = [RegionBasedHaplotypes.overlap_length_between_ranges(region, read_range)
                                   for region in possible_regions]
            max_length = max(overlapping_lengths)
            if max_length <= 0:
                realigned_reads.append(read)
                continue

            max_window_index = max(range(len(possible_regions)), key=lambda i: overlapping_lengths[i])
            assembly_active_regions[max_window_index].assign_read(read)

        for active_region in assembly_active_regions:
            realigned_reads.extend(self.perform_local_alignment(active_region))

        return realigned_reads
