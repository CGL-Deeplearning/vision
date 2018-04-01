import numpy as np
from scipy import misc
from modules.handlers.ImageChannels import imageChannels

"""
This script creates pileup images given a vcf record, bam alignment file and reference fasta file.

imageChannels: Handles how many channels to create for each base and their structure

"""

MAX_COLOR_VALUE = 254.0
BASE_QUALITY_CAP = 40.0
MAP_QUALITY_CAP = 60.0
MAP_QUALITY_FILTER = 5.0
MIN_DELETE_QUALITY = 20.0
MATCH_CIGAR_CODE = 0
INSERT_CIGAR_CODE = 1
DELETE_CIGAR_CODE = 2
IMAGE_DEPTH_THRESHOLD = 300


class ImageCreator:
    """
    Processes a pileup around a position
    """
    def __init__(self, ref_object, reads, contig, pos, alts):
        """
        Initialize PileupProcessor object with required dictionaries
        :param ref_object: pysam FastaFile object that contains the reference
        :param pileupcolumns: pysam AlignmentFIle.pileup object that contains reads of a position
        :param contig: Contig (ex chr3)
        :param pos: Position in contig
        :param genotype: Genotype
        :param alt: Alternate allele
        """
        self.ref_object = ref_object
        self.reads = reads
        self.contig = contig
        self.pos = pos
        self.alt = alts
        # [genomic_position] = [max_insert_length]
        self.insert_length_dictionary = {}
        # [read_id] = {{genomic_position}->base}
        self.read_dictionary = {}
        # [read_id] = {{genomic_position}->insert_bases}
        self.read_insert_dictionary = {}
        # List of Read ids in a genomic position
        self.reads_aligned_to_pos = []
        # genomic_position_1, genomic_position_2...
        self.position_list = [] # used
        self.leftmost_genomic_position = -1
        self.rightmost_genomic_position = -1
        self.genomic_position_projection = {}
        self.reference_base_projection = {}
        self.ref_sequence = ''
        self.parse_reads()
        self.project_genomic_positions()

    def project_genomic_positions(self):
        """
        Generate reference sequence with inserts based on two dictionaries
        :return:
        """
        if self.leftmost_genomic_position < 0:
            self.leftmost_genomic_position = 0
        if self.rightmost_genomic_position < 0:
            self.rightmost_genomic_position = 0

        # get the reference sequence
        ref_seq, error_val = self.ref_object.get_ref_of_region(self.contig,
                                                    ":"+str(self.leftmost_genomic_position+1)+ "-"
                                                    + str(self.rightmost_genomic_position+1))

        if error_val == 1:
            print("ERROR IN FETCHING REFERENCE: ", self.contig, self.pos, self.alt)

        ref_seq_with_insert = ''
        idx = 0
        for i in range(self.leftmost_genomic_position, self.rightmost_genomic_position+1):
            # projection of genomic position
            self.genomic_position_projection[i] = idx
            # get reference of that position
            self.reference_base_projection[i] = ref_seq[i-self.leftmost_genomic_position]
            ref_seq_with_insert += ref_seq[i-self.leftmost_genomic_position]
            idx += 1
            # if genomic position has insert
            if i in self.insert_length_dictionary:
                # append inserted characters to the reference
                ref_seq_with_insert += (self.insert_length_dictionary[i] * '*')
                idx += self.insert_length_dictionary[i]
        # set the reference sequence
        self.ref_sequence = ref_seq_with_insert

        # return index
        return idx

    def length_of_region(self):
        """
        Return the length of the sequence from left to rightmost genomic position.
        :return:
        """
        length = 0
        for i in range(self.leftmost_genomic_position, self.rightmost_genomic_position):
            length += 1
            if i in self.insert_length_dictionary:
                length += self.insert_length_dictionary[i]
        return length

    def initialize_dictionaries(self, genomic_position, read_id, is_insert):
        """
        Initialize all the dictionaries for a specific position
        :param genomic_position: Genomic position of interest
        :param read_id: Read id for which dictionaries should be initialized
        :param is_insert: If the position is an insert
        :return:
        """
        if self.leftmost_genomic_position < 0 or genomic_position < self.leftmost_genomic_position:
            self.leftmost_genomic_position = genomic_position
        if self.rightmost_genomic_position < 0 or genomic_position > self.rightmost_genomic_position:
            self.rightmost_genomic_position = genomic_position

        if read_id not in self.read_dictionary:
            self.read_dictionary[read_id] = {}
            self.read_dictionary[read_id][genomic_position] = ''

        if is_insert:
            if genomic_position not in self.insert_length_dictionary:
                self.insert_length_dictionary[genomic_position] = 0
            if read_id not in self.read_insert_dictionary:
                self.read_insert_dictionary[read_id] = {}
                self.read_insert_dictionary[read_id][genomic_position] = ''

    def save_info_of_a_position(self, genomic_position, read_id, base, base_qual, map_qual, is_rev, cigar_code, is_in):
        """
        Given the attributes of a base at a position
        :param genomic_position: Genomic position
        :param read_id: Read id of a read
        :param base: Base at the position
        :param base_qual: Base quality
        :param map_qual: Map quality
        :param is_rev: If read is reversed
        :param is_in:
        :return:
        """
        self.initialize_dictionaries(genomic_position, read_id, is_in)

        if is_in is False:
            self.read_dictionary[read_id][genomic_position] = (base, base_qual, map_qual, cigar_code, is_rev)
        else:
            self.read_insert_dictionary[read_id][genomic_position] = (base, base_qual, map_qual, cigar_code, is_rev)
            self.insert_length_dictionary[genomic_position] = max(self.insert_length_dictionary[genomic_position],
                                                                  len(base))

    @staticmethod
    def get_attributes_to_save_indel(position, read, read_qs, read_seq):
        return position, \
               read.query_name, \
               read_seq, \
               read_qs, \
               read.mapping_quality, \
               read.is_reverse, \
               INSERT_CIGAR_CODE

    @staticmethod
    def get_attributes_to_save(position, start_pos, read, read_qs, read_seq, is_del):
        if is_del:
            return position, \
                   read.query_name,\
                   '*', \
                   MIN_DELETE_QUALITY, \
                   read.mapping_quality, \
                   read.is_reverse, \
                   DELETE_CIGAR_CODE
        else:
            return position, \
                   read.query_name, \
                   read_seq[position-start_pos], \
                   read_qs[position-start_pos], \
                   read.mapping_quality, \
                   read.is_reverse, \
                   MATCH_CIGAR_CODE

    @staticmethod
    def save_image_as_png(pileup_array, save_dir, file_name):
        pileupArray2d = pileup_array.reshape((pileup_array.shape[0], -1))
        misc.imsave(save_dir + file_name + ".png", pileupArray2d, format="PNG")

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

    def parse_match(self, alignment_position, length, read_sequence, read, read_qs):
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
            gen_pos, read_id, base, base_q, map_q, is_rev, cigar_code = \
                self.get_attributes_to_save(i, start, read, read_qs, read_sequence, False)
            self.save_info_of_a_position(gen_pos, read_id, base, base_q, map_q, is_rev, cigar_code, is_in=False)

    def parse_delete(self, alignment_position, length, read):
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
        for i in range(start, stop):
            gen_pos, read_id, base, base_q, map_q, is_rev, cigar_code = \
                self.get_attributes_to_save(i, start, read, [], '', True)
            self.save_info_of_a_position(gen_pos, read_id, base, base_q, map_q, is_rev, cigar_code, is_in=False)

    def parse_insert(self, alignment_position, read_sequence, read, read_qs):
        """
        Process a cigar operation where there is an insert
        :param alignment_position: Position where the insert happened
        :param read_sequence: The insert read sequence
        :return:

        This method updates the candidates dictionary. Mostly by adding read IDs to the specific positions.
        """
        gen_pos, read_id, base, base_qual, map_qual, is_rev, cigar_code = \
            self.get_attributes_to_save_indel(alignment_position, read, read_qs, read_sequence)

        self.save_info_of_a_position(gen_pos, read_id, base, base_qual, map_qual, is_rev, cigar_code, is_in=True)

    def parse_cigar_tuple(self, cigar_code, length, alignment_position, read_sequence, read, read_qs):
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
                             read=read,
                             read_qs=read_qs)
        elif cigar_code == 1:
            # insert
            # alignment position is where the next alignment starts, for insert and delete this
            # position should be the anchor point hence we use a -1 to refer to the anchor point
            self.parse_insert(alignment_position=alignment_position-1,
                              read_sequence=read_sequence,
                              read=read,
                              read_qs=read_qs)
            ref_index_increment = 0
        elif cigar_code == 2 or cigar_code == 3:
            # delete or ref_skip
            # alignment position is where the next alignment starts, for insert and delete this
            # position should be the anchor point hence we use a -1 to refer to the anchor point
            self.parse_delete(alignment_position=alignment_position-1,
                              length=length,
                              read=read)
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

    def find_read_candidates(self, read):
        """
        This method finds candidates given a read. We walk through the cigar string to find these candidates.
        :param read: Read from which we need to find the variant candidate positions.
        :return:

        Read candidates use a set data structure to find all positions in the read that has a possible variant.
        """
        ref_alignment_start = read.reference_start
        ref_alignment_stop = self.get_read_stop_position(read)

        if self.leftmost_genomic_position == -1 or ref_alignment_start < self.leftmost_genomic_position:
            self.leftmost_genomic_position = ref_alignment_start

        if self.rightmost_genomic_position == -1 or ref_alignment_stop > self.rightmost_genomic_position:
            self.rightmost_genomic_position = ref_alignment_stop

        cigar_tuples = read.cigartuples
        read_sequence = read.query_sequence
        read_qualities = read.query_qualities

        # read_index: index of read sequence
        # ref_index: index of reference sequence
        read_index = 0
        ref_index = 0
        # we don't wat leading indels in a read
        found_valid_cigar = False

        for cigar in cigar_tuples:
            cigar_code = cigar[0]
            length = cigar[1]

            if (cigar_code == 1 or cigar_code == 2) and found_valid_cigar is False:
                read_index += length
                continue

            found_valid_cigar = True
            # get the sequence segments that are effected by this operation
            read_sequence_segment = read_sequence[read_index:read_index+length]
            read_qualities_segment = read_qualities[read_index:read_index + length]

            # send the cigar tuple to get attributes we got by this operation
            ref_index_increment, read_index_increment = \
                self.parse_cigar_tuple(cigar_code=cigar_code,
                                       length=length,
                                       alignment_position=ref_alignment_start+ref_index,
                                       read_sequence=read_sequence_segment,
                                       read_qs=read_qualities_segment,
                                       read=read)

            # increase the read index iterator
            read_index += read_index_increment
            ref_index += ref_index_increment

    def parse_reads(self):
        for i, read in enumerate(self.reads):
            # check if the read is usable
            if read.mapping_quality >= MAP_QUALITY_FILTER and read.is_secondary is False \
                    and read.is_supplementary is False and read.is_unmapped is False:
                self.find_read_candidates(read=read)
                self.reads_aligned_to_pos.append(read.query_name)
            if i > IMAGE_DEPTH_THRESHOLD:
                break

    def check_for_support(self, read_id, ref, alt, poi):
        genomic_start_position = poi
        genomic_end_position = poi + len(ref)
        allele = ''
        for pos in range(genomic_start_position, genomic_end_position):
            if pos in self.read_dictionary[read_id]:
                allele += self.read_dictionary[read_id][pos][0]
            if len(alt) > 1 and read_id in self.read_insert_dictionary and pos in self.read_insert_dictionary[read_id]:
                allele += self.read_insert_dictionary[read_id][pos][0]
        allele = allele.replace('*', '')
        alt = alt.replace('*', '')
        if allele == alt:
            return True
        return False

    def get_row(self, read_id, poi, ref, alts):
        read_list = {}
        read_insert_list = {}
        is_supporting = False
        for alt in alts:
            if self.check_for_support(read_id, ref, alt, poi) is True:
                is_supporting = True
                break

        aligned_positions = sorted(self.read_dictionary[read_id].keys())
        for pos in aligned_positions:
            read_list[pos] = []
            read_list[pos].append(self.read_dictionary[read_id][pos])

            if pos in self.insert_length_dictionary.keys() and self.insert_length_dictionary[pos] > 0:
                read_insert_list[pos] = []
                inserted_bases = 0
                if read_id in self.read_insert_dictionary and pos in self.read_insert_dictionary[read_id]:
                    inserted_bases = len(self.read_insert_dictionary[read_id][pos][0])
                    read_insert_list[pos].append(self.read_insert_dictionary[read_id][pos])

                for i in range(inserted_bases, self.insert_length_dictionary[pos]):
                    read_attribute_tuple = ('*', [BASE_QUALITY_CAP], self.read_dictionary[read_id][pos][2],
                                            INSERT_CIGAR_CODE, self.read_dictionary[read_id][pos][4])
                    read_insert_list[pos].append(read_attribute_tuple)
        return read_list, read_insert_list, is_supporting

    # TEST FIVE CHANNELS
    def get_reference_row(self, image_width):
        image_row = [imageChannels.get_empty_channels() for i in range(image_width)]
        for i in range(0, min(len(self.ref_sequence), image_width)):
            image_row[i] = imageChannels.get_channels_for_ref(self.ref_sequence[i])
        return image_row

    @staticmethod
    def add_empty_rows(image, empty_rows_to_add, image_width):
        for i in range(empty_rows_to_add):
            image.append([imageChannels.get_empty_channels() for i in range(image_width)])
        return image

    def create_image(self, query_pos, ref, alt, image_height=300, image_width=300, ref_band=5):
        whole_image = []
        for i in range(ref_band):
            whole_image.append(self.get_reference_row(image_width))

        for read_id in self.reads_aligned_to_pos:
            row_list, row_insert_list, is_supporting = self.get_row(read_id, query_pos, ref, alt)

            image_row = [imageChannels.get_empty_channels() for i in range(image_width)]

            filter_row = False
            for position in sorted(row_list):
                try:
                    if row_list[position][0][2] < MAP_QUALITY_FILTER:
                        filter_row = True
                        break
                except:
                    print("ERROR IN POSITION: ", position, read_id)


                imagechannels_object = imageChannels(row_list[position][0], self.reference_base_projection[position],
                                                     is_supporting)

                if self.genomic_position_projection[position] < image_width:
                    image_row[self.genomic_position_projection[position]] = imagechannels_object.get_channels()

                if position in row_insert_list.keys():
                    insert_ref = 0
                    for bases in row_insert_list[position]:
                        for base_idx in range(len(bases[0])):
                            insert_ref += 1
                            attribute_tuple = (bases[0][base_idx], bases[1][base_idx], bases[2], bases[3], bases[4])
                            imagechannels_object = imageChannels(attribute_tuple, '*', is_supporting)

                            if self.genomic_position_projection[position] + insert_ref < image_width:
                                image_row[self.genomic_position_projection[position] + insert_ref] = \
                                    imagechannels_object.get_channels()

            if filter_row is False and len(whole_image) < image_height:
                whole_image.append(image_row)

        whole_image = self.add_empty_rows(whole_image, image_height - len(whole_image), image_width)

        image_array = np.array(whole_image).astype(np.uint8)
        return image_array, image_array.shape
