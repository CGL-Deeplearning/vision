from collections import defaultdict
from modules.handlers.ImageChannels import imageChannels
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
"""

DEFAULT_MIN_MAP_QUALITY = 5
MIN_MISMATCH_THRESHOLD = 2
MIN_MISMATCH_PERCENT_THRESHOLD = 2
MIN_COVERAGE_THRESHOLD = 5

PLOIDY = 2
MATCH_ALLELE = 0
MISMATCH_ALLELE = 1
INSERT_ALLELE = 2
DELETE_ALLELE = 3
MIN_DELETE_QUALITY = 20


class CandidateFinder:
    """
    Given reads that align to a site and a pointer to the reference fasta file handler,
    candidate finder finds possible variant candidates_by_read of that site.
    """
    def __init__(self, reads, fasta_handler, chromosome_name, region_start_position, region_end_position, preprocess=True):
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
        self.mismatch_count = defaultdict(int)
        self.match_count = defaultdict(int)

        # the base and the insert dictionary for finding alleles
        self.positional_allele_dictionary = {}
        self.read_allele_dictionary = {}
        self.reference_dictionary = {}

        # few new dictionaries for image creation
        self.base_dictionary = defaultdict(lambda: defaultdict(int))
        self.insert_dictionary = defaultdict(lambda: defaultdict(int))
        self.read_info = defaultdict(list)
        self.insert_length_info = defaultdict(int)
        self.positional_read_info = defaultdict(list)

        # for image generation
        self.image_row_for_reads = defaultdict(list)
        self.image_row_for_ref = defaultdict(list)
        self.positional_info_index_to_position = defaultdict(int)
        self.positional_info_position_to_index = defaultdict(int)
        self.allele_dictionary = defaultdict(lambda: defaultdict(list))
        self.read_id_by_position = defaultdict(list)

        self.preprocess = preprocess

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

    def _update_base_dictionary(self, read_id, pos, base, quality):
        self.base_dictionary[read_id][pos] = (base, quality)

    def _update_insert_dictionary(self, read_id, pos, bases, qualities):
        self.insert_dictionary[read_id][pos] = (bases, qualities)
        self.insert_length_info[pos] = max(self.insert_length_info[pos], len(bases))

    def _update_read_allele_dictionary(self, read_id, pos, allele, type):
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

    def _update_positional_allele_dictionary(self, read_id, pos, allele, type, mapping_quality):
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
        self.allele_dictionary[read_id][pos].append((allele, type))

    def parse_match(self, read_id, alignment_position, length, read_sequence, ref_sequence, qualities):
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
            self._update_base_dictionary(read_id, i, allele, qualities[i-alignment_position])
            if allele != ref:
                self.mismatch_count[i] += 1
                self._update_read_allele_dictionary(read_id, i, allele, MISMATCH_ALLELE)
            else:
                self.match_count[i] += 1
                # this slows things down a lot. Don't add reference allele to the dictionary if we don't use them
                # self._update_read_allele_dictionary(i, allele, MATCH_ALLELE)

    def parse_delete(self, read_id, alignment_position, length, ref_sequence):
        """
        Process a cigar operation that is a delete
        :param alignment_position: Alignment position
        :param length: Length of the delete
        :param ref_sequence: Reference sequence of delete
        :return:

        This method updates the candidates dictionary.
        """
        # actual delete position starts one after the anchor
        start = alignment_position + 1
        stop = start + length
        self.mismatch_count[alignment_position] += 1

        for i in range(start, stop):
            self._update_base_dictionary(read_id, i, '*', MIN_DELETE_QUALITY)
            # increase the coverage
            self.mismatch_count[i] += 1
            self.coverage[i] += 1

        # the allele is the anchor + what's being deleted
        allele = self.reference_dictionary[alignment_position] + ref_sequence

        # record the delete where it first starts
        self._update_read_allele_dictionary(read_id, alignment_position + 1, allele, DELETE_ALLELE)

    def parse_insert(self, read_id, alignment_position, read_sequence, qualities):
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
        self._update_read_allele_dictionary(read_id, alignment_position + 1, allele, INSERT_ALLELE)
        self._update_insert_dictionary(read_id, alignment_position, read_sequence, qualities)

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
        # if the region has very high coverage, we are not going to parse through all the reads
        if self.coverage[ref_alignment_start] > 300:
            return False
        cigar_tuples = read.cigartuples
        read_sequence = read.query_sequence
        read_id = read.query_name
        read_quality = read.query_qualities
        ref_sequence = self.fasta_handler.get_sequence(chromosome_name=self.chromosome_name,
                                                       start=ref_alignment_start,
                                                       stop=ref_alignment_stop+10)

        self.read_info[read_id] = (ref_alignment_start, ref_alignment_stop, read.mapping_quality, read.is_reverse)
        for pos in range(ref_alignment_start, ref_alignment_stop):
            self.read_id_by_position[pos].append((read_id, ref_alignment_start, ref_alignment_stop))
        for i, ref_base in enumerate(ref_sequence):
            self._update_reference_dictionary(ref_alignment_start + i, ref_base)

        # read_index: index of read sequence
        # ref_index: index of reference sequence
        read_index = 0
        ref_index = 0
        found_valid_cigar = False
        for cigar in cigar_tuples:
            cigar_code = cigar[0]
            length = cigar[1]

            # get the sequence segments that are affected by this operation
            ref_sequence_segment = ref_sequence[ref_index:ref_index+length]
            read_quality_segment = read_quality[read_index:read_index+length]
            read_sequence_segment = read_sequence[read_index:read_index+length]

            if cigar_code != 0 and found_valid_cigar is False:
                read_index += length
                continue
            found_valid_cigar = True

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

        # after collecting all alleles from reads, update the global dictionary
        for position in self.read_allele_dictionary.keys():
            if position < self.region_start_position or position > self.region_end_position:
                continue



            for record in self.read_allele_dictionary[position]:
                # there can be only one record per position in a read
                allele, allele_type = record

                if allele_type == MATCH_ALLELE or allele_type == MISMATCH_ALLELE:
                    # If next allele is indel then group it with the current one, don't make a separate one
                    if position + 1 <= ref_alignment_stop and position + 1 in self.read_allele_dictionary.keys():
                        next_allele, next_allele_type = list(self.read_allele_dictionary[position + 1].keys())[0]
                        if next_allele_type == INSERT_ALLELE or next_allele_type == DELETE_ALLELE:
                            continue
                    self.positional_read_info[position].append(
                        (read_id, ref_alignment_start, ref_alignment_stop, read.mapping_quality))
                    self._update_positional_allele_dictionary(read_id, position, allele, allele_type,
                                                              read.mapping_quality)
                else:
                    # it's an insert or delete, so, add to the previous position
                    self.positional_read_info[position-1].append(
                        (read_id, ref_alignment_start, ref_alignment_stop, read.mapping_quality))
                    self._update_positional_allele_dictionary(read_id, position-1, allele, allele_type,
                                                              read.mapping_quality)
        return True

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
            self.parse_match(read_id=read_id,
                             alignment_position=alignment_position,
                             length=length,
                             read_sequence=read_sequence,
                             ref_sequence=ref_sequence,
                             qualities=quality)
        elif cigar_code == 1:
            # insert
            # alignment position is where the next alignment starts, for insert and delete this
            # position should be the anchor point hence we use a -1 to refer to the anchor point
            self.parse_insert(read_id=read_id,
                              alignment_position=alignment_position-1,
                              read_sequence=read_sequence,
                              qualities=quality)
            ref_index_increment = 0
        elif cigar_code == 2 or cigar_code == 3:
            # delete or ref_skip
            # alignment position is where the next alignment starts, for insert and delete this
            # position should be the anchor point hence we use a -1 to refer to the anchor point
            self.parse_delete(read_id=read_id,
                              alignment_position=alignment_position-1,
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

    # def _get_record(self, pos, alt1, alt2, ref, ref_count):
    #     """
    #     Given alternate alleles, return a record that we can save in bed file
    #     :param pos: Genomic position
    #     :param alt1: alternate allele 1
    #     :param alt2: alternate allele 2
    #     :param ref: reference sequence
    #     :return: A record to be saved in bed file
    #     """
    #     alt1_tuple, alt1_count, alt1_freq = alt1
    #     alt1_seq, alt1_type = alt1_tuple
    #     if alt2 == '.':
    #         pos_end = pos + len(alt1_seq) - 1
    #         return [pos, pos_end, ref, alt1_seq, '.', alt1_type, 0, ref_count, alt1_count, 0, alt1_freq, 0]
    #     alt2_tuple, alt2_count, alt2_freq = alt2
    #     alt2_seq, alt2_type = alt2_tuple
    #     pos_end = pos + max(len(alt1_seq), len(alt2_seq)) - 1
    #
    #     return [pos, pos_end, ref, alt1_seq, alt2_seq, alt1_type, alt2_type, ref_count, alt1_count, alt2_count,
    #             alt1_freq, alt2_freq]

    # def postprocess_reads(self, read_id_list):
    #     for read_id in read_id_list:
    #         star_pos, end_pos, mapping_quality, strand_direction = self.read_info[read_id]
    #         read_to_image_row = []
    #         for pos in range(star_pos, end_pos):
    #             if pos not in self.base_dictionary[read_id] and pos not in self.insert_dictionary[read_id]:
    #                 print(pos, read_id)
    #                 continue
    #             if pos in self.base_dictionary[read_id]:
    #                 base, base_q = self.base_dictionary[read_id][pos]
    #                 cigar_code = 0 if base != '*' else 1
    #                 ref_base = self.reference_dictionary[pos]
    #                 pileup_attributes = (base, base_q, mapping_quality, cigar_code, strand_direction)
    #                 channel_object = imageChannels(pileup_attributes, ref_base, 0)
    #                 read_to_image_row.append(channel_object.get_channels_except_support())
    #             if pos in self.insert_length_info:
    #                 length_of_insert = self.insert_length_info[pos]
    #                 total_insert_bases = 0
    #                 if read_id in self.insert_dictionary and pos in self.insert_dictionary[read_id]:
    #                     in_bases, in_qualities = self.insert_dictionary[read_id][pos]
    #                     total_insert_bases = len(in_bases)
    #                     for i in range(total_insert_bases):
    #                         base = in_bases[i]
    #                         base_q = in_qualities[i]
    #                         cigar_code = 2
    #                         ref_base = ''
    #                         pileup_attributes = (base, base_q, mapping_quality, cigar_code, strand_direction)
    #                         channel_object = imageChannels(pileup_attributes, ref_base, 0)
    #                         read_to_image_row.append(channel_object.get_channels_except_support())
    #                 if length_of_insert > total_insert_bases:
    #                     dot_bases = length_of_insert - total_insert_bases
    #                     for i in range(dot_bases):
    #                         base = '*'
    #                         base_q = MIN_DELETE_QUALITY
    #                         cigar_code = 2
    #                         ref_base = ''
    #                         pileup_attributes = (base, base_q, mapping_quality, cigar_code, strand_direction)
    #                         channel_object = imageChannels(pileup_attributes, ref_base, 0)
    #                         read_to_image_row.append(channel_object.get_channels_except_support())
    #
    #         self.image_row_for_reads[read_id] = (read_to_image_row, star_pos, end_pos)

    def get_pileup_dictionaries(self):
        return self.image_row_for_reads, self.image_row_for_ref, self.positional_info_position_to_index, \
               self.positional_info_index_to_position, self.allele_dictionary, self.read_id_by_position

    def postprocess_reference(self):
        left_position = min(self.reference_dictionary.keys())
        right_position = max(self.reference_dictionary.keys()) + 1
        reference_to_image_row = []
        index = 0

        for pos in range(left_position, right_position):
            base = self.reference_dictionary[pos] if pos in self.reference_dictionary else 'N'
            reference_to_image_row.append(imageChannels.get_channels_for_ref(base))
            self.positional_info_index_to_position[index] = (pos, False)
            self.positional_info_position_to_index[pos] = index
            index += 1
            if pos in self.insert_length_info:
                for i in range(self.insert_length_info[pos]):
                    base = '*'
                    reference_to_image_row.append(imageChannels.get_channels_for_ref(base))
                    self.positional_info_index_to_position[index] = (pos, True)
                    index += 1

        self.image_row_for_ref = (reference_to_image_row, left_position, right_position)

    def get_positional_data(self, reads):
        """
        Parse reads to aligned to a site to find variants
        :param reads: Set of reads aligned
        :return:
        """
        st_time = time.time()
        read_id_list = []
        total_reads = 0
        for read in reads:
            # check if the read is usable
            if read.mapping_quality >= DEFAULT_MIN_MAP_QUALITY and read.is_secondary is False \
                    and read.is_supplementary is False and read.is_unmapped is False and read.is_qcfail is False:

                read.query_name = read.query_name + '_1' if read.is_read1 else read.query_name + '_2'
                if self.find_read_candidates(read=read):
                    read_id_list.append(read.query_name)
                    total_reads += 1

        if total_reads == 0:
            return None

        return self.chromosome_name, self.region_start_position, self.region_end_position, self.reference_dictionary, self.positional_allele_dictionary, self.coverage
