from collections import defaultdict
import operator
import time
import math
"""
CandidateFinder finds possible positions based on edits we see in reads.
It parses through each read of a site and finds possible candidate positions.

Dictionaries it updates:
- candidate_by_read:    records at each position what reads had mismatches  {int -> list}
- coverage:             records coverage of a position                      {int -> int}
- edit_count:           records number of mismatches in a position          {int -> int}

Other data structures:
- candidate_positions (set):    set of positions that of potential candidate variants
- merged_windows (list):        list of windows of potential candidate variants.

Merged_windows is constructed from candidate_positions. If two positions fall within
MERGE_WINDOW_DISTANCE we merge them in a single window.
"""
DEFAULT_MIN_MAP_QUALITY = 5
MERGE_WINDOW_DISTANCE = 0
MERGE_WINDOW_OFFSET = 0
MIN_MISMATCH_THRESHOLD = 0
MIN_MISMATCH_PERCENT_THRESHOLD = 0
MIN_COVERAGE_THRESHOLD = 3
PLOIDY = 2
MATCH_ALLELE = 0
MISMATCH_ALLELE = 1
INSERT_ALLELE = 2
DELETE_ALLELE = 3


class CandidateFinder:
    """
    Given reads that align to a site and a pointer to the reference fasta file handler,
    candidate finder finds possible variant candidates_by_read of that site.
    """
    def __init__(self, reads, fasta_handler, chromosome_name, region_start_position, region_end_position):
        """
        Initialize a candidate finder object.
        :param reads: Reads that align to the site
        :param fasta_handler: Reference sequence handler
        :param chromosome_name: Chromosome name
        :param region_start_position: Start position of the region
        :param region_end_position: End position of the region
        """
        self.region_start_position = region_start_position
        self.region_end_position = region_end_position
        self.chromosome_name = chromosome_name
        self.fasta_handler = fasta_handler
        self.reads = reads

        # the store which reads are creating candidates in that position
        self.coverage = defaultdict(int)
        self.rms_mq = defaultdict(int)
        self.mismatch_count = defaultdict(int)
        self.match_count = defaultdict(int)

        # the base and the insert dictionary for finding alleles
        self.positional_allele_dictionary = {}
        self.read_allele_dictionary = {}
        self.reference_dictionary = {}

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
            positions = read.get_reference_positions()

            # find last entry that isn't None
            i = len(positions) - 1
            ref_alignment_stop = positions[-1]
            while i > 0 and ref_alignment_stop is None:
                i -= 1
                ref_alignment_stop = positions[i]

        return ref_alignment_stop

    def _update_read_allele_dictionary(self, pos, allele, type):
        """
        Update the read dictionary with an allele
        :param pos: Genomic position
        :param allele: Allele found in that position
        :param type: IN, DEL or SUB
        :return:
        """
        if pos not in self.read_allele_dictionary:
            self.read_allele_dictionary[pos] = {}
        if (allele, type) not in self.read_allele_dictionary[pos]:
            self.read_allele_dictionary[pos][(allele, type)] = 0

        self.read_allele_dictionary[pos][(allele, type)] += 1

    def _update_positional_allele_dictionary(self, pos, allele, type, mapping_quality):
        """
        Update the positional allele dictionary that contains whole genome allele information
        :param pos: Genomic position
        :param allele: Allele found
        :param type: IN, DEL or SUB
        :param mapping_quality: Mapping quality of the read where the allele was found
        :return:
        """
        if pos not in self.positional_allele_dictionary:
            self.positional_allele_dictionary[pos] = {}
        if (allele, type) not in self.positional_allele_dictionary[pos]:
            self.positional_allele_dictionary[pos][(allele, type)] = 0

        # increase the allele frequency of the allele at that position
        self.positional_allele_dictionary[pos][(allele, type)] += 1

    def parse_match(self, alignment_position, length, read_sequence, ref_sequence):
        """
        Process a cigar operation that is a match
        :param alignment_position: Position where this match happened
        :param read_sequence: Read sequence
        :param ref_sequence: Reference sequence
        :param length: Length of the operation
        :return:

        This method updates the candidates dictionary.
        """
        start = alignment_position
        stop = start + length
        for i in range(start, stop):
            self.coverage[i] += 1
            allele = read_sequence[i-alignment_position]
            ref = ref_sequence[i-alignment_position]

            if allele != ref:
                self.mismatch_count[i] += 1
                self._update_read_allele_dictionary(i, allele, MISMATCH_ALLELE)
            else:
                self.match_count[i] += 1
                # this slows things down a lot. Don't add reference allele to the dictionary if we don't use them
                # self._update_read_allele_dictionary(i, allele, MATCH_ALLELE)

    def parse_delete(self, alignment_position, length, ref_sequence):
        """
        Process a cigar operation that is a delete
        :param alignment_position: Alignment position
        :param length: Length of the delete
        :param ref_sequence: Reference sequence of delete
        :return:

        This method updates the candidates dictionary.
        """
        if alignment_position < self.region_start_position or alignment_position > self.region_end_position:
            return

        # actual delete position starts one after the anchor
        start = alignment_position + 1
        stop = start + length
        self.mismatch_count[alignment_position] += 1
        for i in range(start, stop):
            # increase the coverage
            self.mismatch_count[i] += 1
            self.coverage[i] += 1

        # the allele is the anchor + what's being deleted
        allele = self.reference_dictionary[alignment_position] + ref_sequence

        # record the delete where it first starts
        self._update_read_allele_dictionary(alignment_position + 1, allele, DELETE_ALLELE)

    def parse_insert(self, alignment_position, read_sequence):
        """
        Process a cigar operation where there is an insert
        :param alignment_position: Position where the insert happened
        :param read_sequence: The insert read sequence
        :return:

        This method updates the candidates dictionary. Mostly by adding read IDs to the specific positions.
        """
        # the allele is the anchor + what's being deleted
        allele = self.reference_dictionary[alignment_position] + read_sequence

        # record the insert where it first starts
        self.mismatch_count[alignment_position] += 1
        self._update_read_allele_dictionary(alignment_position + 1, allele, INSERT_ALLELE)

    def _update_reference_dictionary(self, position, ref_base):
        """
        Update the reference dictionary
        :param position: Genomic position
        :param ref_base: Reference base at that position
        :return:
        """
        self.reference_dictionary[position] = ref_base

    def find_read_candidates(self, read):
        """
        This method finds candidates given a read. We walk through the cigar string to find these candidates.
        :param read: Read from which we need to find the variant candidate positions.
        :return:

        Read candidates use a set data structure to find all positions in the read that has a possible variant.
        """
        self.read_allele_dictionary = {}
        ref_alignment_start = read.reference_start
        ref_alignment_stop = self.get_read_stop_position(read)
        cigar_tuples = read.cigartuples
        read_sequence = read.query_sequence
        ref_sequence = self.fasta_handler.get_sequence(chromosome_name=self.chromosome_name,
                                                       start=ref_alignment_start,
                                                       stop=ref_alignment_stop)

        for i, ref_base in enumerate(ref_sequence):
            self._update_reference_dictionary(ref_alignment_start + i, ref_base)

        # read_index: index of read sequence
        # ref_index: index of reference sequence
        read_index = 0
        ref_index = 0

        for cigar in cigar_tuples:
            cigar_code = cigar[0]
            length = cigar[1]
            # get the sequence segments that are effected by this operation
            ref_sequence_segment = ref_sequence[ref_index:ref_index+length]
            read_sequence_segment = read_sequence[read_index:read_index+length]

            # send the cigar tuple to get attributes we got by this operation
            ref_index_increment, read_index_increment = \
                self.parse_cigar_tuple(cigar_code=cigar_code,
                                       length=length,
                                       alignment_position=ref_alignment_start+ref_index,
                                       ref_sequence=ref_sequence_segment,
                                       read_sequence=read_sequence_segment)

            # increase the read index iterator
            read_index += read_index_increment
            ref_index += ref_index_increment

        # after collecting all alleles from reads, update the global dictionary
        for position in self.read_allele_dictionary.keys():
            if position < self.region_start_position or position > self.region_end_position:
                continue
            self.rms_mq[position] += read.mapping_quality * read.mapping_quality
            for record in self.read_allele_dictionary[position]:
                # there can be only one record per position in a read
                allele, allele_type = record
                if allele_type == MATCH_ALLELE or allele_type == MISMATCH_ALLELE:
                    # If next allele is indel then group it with the current one, don't make a separate one
                    if position + 1 <= ref_alignment_stop and position + 1 in self.read_allele_dictionary.keys():
                        next_allele, next_allele_type = list(self.read_allele_dictionary[position + 1].keys())[0]
                        if next_allele_type == INSERT_ALLELE or next_allele_type == DELETE_ALLELE:
                            continue
                    self._update_positional_allele_dictionary(position, allele, allele_type, read.mapping_quality)
                else:
                    # it's an insert or delete, so, add to the previous position
                    self._update_positional_allele_dictionary(position-1, allele, allele_type, read.mapping_quality)

    def parse_cigar_tuple(self, cigar_code, length, alignment_position, ref_sequence, read_sequence):
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
            self.parse_match(alignment_position=alignment_position,
                             length=length,
                             read_sequence=read_sequence,
                             ref_sequence=ref_sequence)
        elif cigar_code == 1:
            # insert
            # alignment position is where the next alignment starts, for insert and delete this
            # position should be the anchor point hence we use a -1 to refer to the anchor point
            self.parse_insert(alignment_position=alignment_position-1,
                              read_sequence=read_sequence)
            ref_index_increment = 0
        elif cigar_code == 2 or cigar_code == 3:
            # delete or ref_skip
            # alignment position is where the next alignment starts, for insert and delete this
            # position should be the anchor point hence we use a -1 to refer to the anchor point
            self.parse_delete(alignment_position=alignment_position-1,
                              ref_sequence=ref_sequence,
                              length=length)
            read_index_increment = 0
        elif cigar_code == 4:
            # soft clip
            ref_index_increment = 0
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
            raise("INVALID CIGAR CODE: %s" % cigar_code)

        return ref_index_increment, read_index_increment

    def _filter_alleles(self, position, allele_frequency_list):
        """
        Apply filter to alleles. The filter we use now are:
        MIN_MISMATCH_THRESHOLD: The count of the allele has to be greater than this value
        MIN_MISMATCH_PERCENT_THRESHOLD: The percent to the coverage has to be greater than this
        MIN_COVERAGE_THRESHOLD: Coverage of the threshold has to be greater than this
        :param position: genomic position
        :param allele_frequency_list: list of tuples containing allele sequence and their frequency
        :return: A filtered list of alleles
        """
        filtered_list = list()
        for allele, count in allele_frequency_list:
            coverage = self.coverage[position] if self.coverage[position] else 0
            frequency = round(count / self.coverage[position], 3) if self.coverage[position] else 0
            if count > MIN_MISMATCH_THRESHOLD and frequency * 100 > MIN_MISMATCH_PERCENT_THRESHOLD\
                    and coverage > MIN_COVERAGE_THRESHOLD:
                filtered_list.append((allele, count, frequency))
        return filtered_list

    def _get_record(self, pos, alt1, alt2, ref, ref_count):
        """
        Given alternate alleles, return a record that we can save in bed file
        :param pos: Genomic position
        :param alt1: alternate allele 1
        :param alt2: alternate allele 2
        :param ref: reference sequence
        :return: A record to be saved in bed file
        """
        alt1_tuple, alt1_count, alt1_freq = alt1
        alt1_seq, alt1_type = alt1_tuple
        pos_end = pos
        if alt2 == '.':
            if alt1_type == DELETE_ALLELE:
                ref, alt1_seq = alt1_seq, ref
                pos_end = pos + len(ref) - 1
            return [pos, pos_end, ref, alt1_seq, '.', alt1_type, 0, ref_count, alt1_count, 0, alt1_freq, 0]
        alt2_tuple, alt2_count, alt2_freq = alt2
        alt2_seq, alt2_type = alt2_tuple
        pos_end = pos
        if alt1_type == DELETE_ALLELE and alt2_type == DELETE_ALLELE:
            if len(alt1_seq) < len(alt2_seq):
                alt1_seq, alt2_seq = alt2_seq, alt1_seq
                alt1_count, alt2_count = alt2_count, alt1_count
                alt1_freq, alt2_freq = alt2_freq, alt1_freq
            ref, alt1_seq = alt1_seq, ref
            pos_end = pos + len(ref) - 1
        elif alt1_type == DELETE_ALLELE:
            ref, alt1_seq = alt1_seq, ref
            alt2_seq = alt2_seq + ref[1:]
            pos_end = pos + len(ref) - 1
        elif alt2_type == DELETE_ALLELE:
            ref, alt2_seq = alt2_seq, ref
            alt1_seq = alt1_seq + ref[1:]
            pos_end = pos + len(ref) - 1

        return [pos, pos_end, ref, alt1_seq, alt2_seq, alt1_type, alt2_type, ref_count, alt1_count, alt2_count, alt1_freq, alt2_freq]

    def parse_reads_and_select_candidates(self, reads):
        """
        Parse reads to aligned to a site to find variants
        :param reads: Set of reads aligned
        :return:
        """
        for read in reads:
            # check if the read is usable
            if read.mapping_quality >= DEFAULT_MIN_MAP_QUALITY and read.is_secondary is False \
                    and read.is_supplementary is False and read.is_unmapped is False:
                self.find_read_candidates(read=read)
        # print('Read processing time', time.time()-start_time)
        selected_allele_list = []
        for pos in range(self.region_start_position, self.region_end_position):
            if pos not in self.positional_allele_dictionary:
                continue

            ref = self.reference_dictionary[pos]

            all_allele_dictionary = self.positional_allele_dictionary[pos]
            all_mismatch_count = 0
            for allele in all_allele_dictionary:
                all_mismatch_count += all_allele_dictionary[allele]

            # pick the top 2 most frequent allele
            allele_frequency_list = list(sorted(all_allele_dictionary.items(), key=operator.itemgetter(1), reverse=True))[:PLOIDY]

            allele_list = self._filter_alleles(pos, allele_frequency_list)

            alt1 = allele_list[0] if len(allele_list) >= 1 else None
            alt2 = allele_list[1] if len(allele_list) >= 2 else '.'
            if alt1 is None:
                continue
            mq_rms = round(math.sqrt(self.rms_mq[pos]/self.coverage[pos]), 3) if self.coverage[pos] > 0 else 0
            dp = self.coverage[pos]
            ref_count = self.coverage[pos] - all_mismatch_count
            candidate_record = [self.chromosome_name] + self._get_record(pos, alt1, alt2, ref, ref_count) + [mq_rms] + [dp]
            selected_allele_list.append(candidate_record)

        return selected_allele_list
