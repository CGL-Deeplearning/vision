from collections import defaultdict

"""doing this: https://software.broadinstitute.org/gatk/documentation/article.php?id=11077"""

MATCH_WEIGHT = -0.06
MISMATCH_WEIGHT = -0.09
INSERT_WEIGHT = 2.5
DELETE_WEIGHT = 1.8
SOFT_CLIP_WEIGHT = 3.0
THRESHOLD_VALUE = 2.2

MIN_REGION_SIZE = 80


class ActiveRegionFinder:

    def __init__(self, fasta_handler, chromosome_name, region_start, region_end):
        self.candidate_position_weighted_sum = defaultdict(int)
        self.candidate_positions = defaultdict(int)
        self.fasta_handler = fasta_handler
        self.chromosome_name = chromosome_name
        self.region_start_position = region_start
        self.region_end_position = region_end

    @staticmethod
    def get_read_stop_position(read):
        """
        Returns the stop position of the reference to where the read stops aligning
        :param read: The read
        :return: stop position of the reference where the read last aligned
        """
        ref_alignment_stop = read.reference_end

        # only find the position if the reference end is fetched as none from pysam API
        if ref_alignment_stop is None:
            positions = read.reference_positions

            # find last entry that isn't None
            i = len(positions) - 1
            ref_alignment_stop = positions[-1]
            while i > 0 and ref_alignment_stop is None:
                i -= 1
                ref_alignment_stop = positions[i]

        return ref_alignment_stop

    def find_read_candidates(self, read):
        """
        This method finds candidates given a read. We walk through the cigar string to find these candidates.
        :param read: Read from which we need to find the variant candidate positions.
        :return:

        Read candidates use a set data structure to find all positions in the read that has a possible variant.
        """
        ref_alignment_start = read.reference_start
        cigar_tuples = read.cigartuples
        read_sequence = read.query_sequence
        read_id = read.query_name
        read_quality = read.query_qualities
        ref_alignment_stop = self.get_read_stop_position(read)
        read.reference_end = ref_alignment_stop
        ref_sequence = self.fasta_handler.get_sequence(chromosome_name=self.chromosome_name,
                                                       start=ref_alignment_start,
                                                       stop=ref_alignment_stop+10)
        # read_index: index of read sequence
        # ref_index: index of reference sequence
        read_index = 0
        ref_index = 0

        for cigar in cigar_tuples:
            cigar_code = cigar[0]
            length = cigar[1]
            # get the sequence segments that are effected by this operation
            ref_sequence_segment = ref_sequence[ref_index:ref_index+length]
            read_quality_segment = read_quality[read_index:read_index+length]
            read_sequence_segment = read_sequence[read_index:read_index+length]

            # send the cigar tuple to get attributes we got by this operation
            ref_index_increment, read_index_increment = \
                self.parse_cigar_tuple(cigar_code=cigar_code,
                                       length=length,
                                       alignment_position=ref_alignment_start+ref_index,
                                       ref_sequence=ref_sequence_segment,
                                       read_sequence=read_sequence_segment,
                                       read_id=read_id,
                                       quality=read_quality_segment)

            # increase the read index iterator
            read_index += read_index_increment
            ref_index += ref_index_increment

    def parse_cigar_tuple(self, cigar_code, length, alignment_position, ref_sequence, read_sequence, read_id, quality):
        """
        Parse through a cigar operation to find possible candidate variant positions in the read
        :param cigar_code: Cigar operation code
        :param length: Length of the operation
        :param alignment_position: Alignment position corresponding to the reference
        :param ref_sequence: Reference sequence
        :param read_sequence: Read sequence
        :return:

        cigar key map based on operation.
        details: http://pysam.readthedocs.io/en/latest/api.html#pysam.AlignedSegment.cigartuples
        0: "MATCH",
        1: "INSERT",
        2: "DELETE",
        3: "REFSKIP",
        4: "SOFTCLIP",
        5: "HARDCLIP",
        6: "PAD"
        """
        # get what kind of code happened
        ref_index_increment = length
        read_index_increment = length

        # deal different kinds of operations
        if cigar_code == 0:
            # match
            start = alignment_position
            stop = start + length
            for i in range(start, stop):
                allele = read_sequence[i - alignment_position]
                ref = ref_sequence[i - alignment_position]
                # self._update_base_dictionary(read_id, i, allele, qualities[i-alignment_position])
                if allele != ref:
                    self.candidate_position_weighted_sum[i] += MISMATCH_WEIGHT
                else:
                    self.candidate_position_weighted_sum[i] += MATCH_WEIGHT
        elif cigar_code == 1:
            # insert
            # alignment position is where the next alignment starts, for insert and delete this
            # position should be the anchor point hence we use a -1 to refer to the anchor point
            start = alignment_position - length + 1
            end = alignment_position + length
            for pos in range(start, end+1):
                self.candidate_position_weighted_sum[pos] += INSERT_WEIGHT
            ref_index_increment = 0
        elif cigar_code == 2 or cigar_code == 3:
            # delete or ref_skip
            # alignment position is where the next alignment starts, for insert and delete this
            # position should be the anchor point hence we use a -1 to refer to the anchor point
            start = alignment_position + 1
            end = alignment_position + length
            for pos in range(start, end + 1):
                self.candidate_position_weighted_sum[pos] += DELETE_WEIGHT
            read_index_increment = 0
        elif cigar_code == 4:
            # soft clip
            ref_index_increment = 0
            start = alignment_position - length + 1
            end = alignment_position + length
            for pos in range(start, end + 1):
                self.candidate_position_weighted_sum[pos] += INSERT_WEIGHT
            # print("CIGAR CODE ERROR SC")
        elif cigar_code == 5:
            # hard clip
            ref_index_increment = 0
            read_index_increment = 0
            # print("CIGAR CODE ERROR HC")
        elif cigar_code == 6:
            # pad
            ref_index_increment = 0
            read_index_increment = 0
            # print("CIGAR CODE ERROR PAD")
        else:
            raise("INVALID CIGAR CODE: %s THIS SHOULD NEVER HAPPEN" % cigar_code)

        return ref_index_increment, read_index_increment

    @staticmethod
    def create_active_regions(candidate_positions):
        windows = []

        start_pos, end_pos = None, None
        for pos in sorted(candidate_positions):
            if start_pos is None:
                start_pos = pos
                end_pos = pos
            elif pos > end_pos + MIN_REGION_SIZE:
                windows.append((start_pos - MIN_REGION_SIZE, end_pos + MIN_REGION_SIZE))
                start_pos = pos
                end_pos = pos
            else:
                end_pos = pos
        if start_pos is not None:
            windows.append((start_pos - MIN_REGION_SIZE, end_pos + MIN_REGION_SIZE))

        return sorted(windows, key=lambda tup: (tup[0], tup[1]))

    def select_active_region(self, reads):
        if not reads:
            return []

        for read in reads:
            self.find_read_candidates(read)

        candidate_positions = []
        for pos in range(self.region_start_position, self.region_end_position):
            if self.candidate_position_weighted_sum[pos] > THRESHOLD_VALUE:
                candidate_positions.append(pos)
        active_regions = self.create_active_regions(candidate_positions)
        return active_regions
