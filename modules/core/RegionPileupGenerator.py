from collections import defaultdict
from modules.handlers.BamHandler import BamHandler
from modules.handlers.FastaHandler import FastaHandler
from modules.handlers.ImageChannels_seq2seq import ImageChannels
from scipy import misc
import sys
import numpy as np
from modules.handlers.TextColor import TextColor

DEFAULT_MIN_MAP_QUALITY = 1
DEBUG_MESSAGE = False
MIN_DELETE_QUALITY = 20
VCF_INDEX_BUFFER = -1
ALLELE_FREQUENCY_THRESHOLD_FOR_REPORTING = 0.2
WARN_COLOR = TextColor.RED
PLOIDY = 2
MATCH_ALLELE = 0
MISMATCH_ALLELE = 1
INSERT_ALLELE = 2
DELETE_ALLELE = 3


class RegionPileupGenerator:
    def __init__(self, bam_file, ref_file, vcf_file, chr_name):
        self.bam_handler = BamHandler(bam_file)
        self.fasta_handler = FastaHandler(ref_file)
        self.vcf_path = vcf_file

        self.chromosome_name = chr_name

        # the store which reads are creating candidates in that position
        self.coverage = defaultdict(int)
        self.rms_mq = defaultdict(int)
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
        self.vcf_positional_dict = defaultdict(list)

        # for image generation
        self.image_row_for_reads = defaultdict(list)
        self.image_row_for_ref = defaultdict(list)
        self.positional_info_index_to_position = defaultdict(int)
        self.positional_info_position_to_index = defaultdict(int)
        self.allele_dictionary = defaultdict(lambda: defaultdict(list))
        self.read_id_by_position = defaultdict(list)
        self.base_frequency = defaultdict(lambda: defaultdict(int))
        self.index_based_coverage = defaultdict(int)
        self.reference_base_by_index = defaultdict(int)

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

    def _update_reference_dictionary(self, position, ref_base):
        """
        Update the reference dictionary
        :param position: Genomic position
        :param ref_base: Reference base at that position
        :return:
        """
        self.reference_dictionary[position] = ref_base

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

    def process_read(self, read, interval_start, interval_end):
        """
        This method converts a read to a
        :param read: Read from which we need to find the variant candidate positions.
        :return:

        Read candidates use a set data structure to find all positions in the read that has a possible variant.
        """
        self.read_allele_dictionary = {}
        ref_alignment_start = read.reference_start
        ref_alignment_stop = self.get_read_stop_position(read)
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
            # get the sequence segments that are effected by this operation
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
            if position < interval_start or position > interval_end:
                continue
            self.rms_mq[position] += read.mapping_quality * read.mapping_quality
            for record in self.read_allele_dictionary[position]:
                # there can be only one record per position in a read
                allele, allele_type = record

                if allele_type == MISMATCH_ALLELE:
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
                    self.positional_read_info[position - 1].append(
                        (read_id, ref_alignment_start, ref_alignment_stop, read.mapping_quality))
                    self._update_positional_allele_dictionary(read_id, position - 1, allele, allele_type,
                                                              read.mapping_quality)

    def postprocess_reads(self, read_id_list):
        for read_id in read_id_list:
            star_pos, end_pos, mapping_quality, strand_direction = self.read_info[read_id]
            read_to_image_row = []
            for pos in range(star_pos, end_pos):
                if pos not in self.base_dictionary[read_id] and pos not in self.insert_dictionary[read_id]:
                    print(pos, read_id)
                    continue
                if pos in self.base_dictionary[read_id]:
                    base, base_q = self.base_dictionary[read_id][pos]

                    # print(self.positional_info_position_to_index[pos], base)
                    # print(self.base_frequency[self.positional_info_position_to_index[pos]][base])
                    # exit()
                    cigar_code = 0 if base != '*' else 1
                    ref_base = self.reference_dictionary[pos]
                    pileup_attributes = (base, base_q, mapping_quality, cigar_code, strand_direction)
                    channel_object = ImageChannels(pileup_attributes, ref_base, 0)
                    read_to_image_row.append(channel_object.get_channels_except_support())
                    self.index_based_coverage[self.positional_info_position_to_index[pos]] += 1
                    if base == '*':
                        self.base_frequency[self.positional_info_position_to_index[pos]]['.'] += 1
                    else:
                        self.base_frequency[self.positional_info_position_to_index[pos]][base] += 1
                if pos in self.insert_length_info:
                    length_of_insert = self.insert_length_info[pos]
                    total_insert_bases = 0
                    if read_id in self.insert_dictionary and pos in self.insert_dictionary[read_id]:
                        in_bases, in_qualities = self.insert_dictionary[read_id][pos]
                        total_insert_bases = len(in_bases)
                        for i in range(total_insert_bases):
                            base = in_bases[i]
                            base_q = in_qualities[i]
                            cigar_code = 2
                            ref_base = ''
                            pileup_attributes = (base, base_q, mapping_quality, cigar_code, strand_direction)
                            channel_object = ImageChannels(pileup_attributes, ref_base, 0)
                            read_to_image_row.append(channel_object.get_channels_except_support())

                            self.base_frequency[self.positional_info_position_to_index[pos] + i + 1][base] += 1
                            self.index_based_coverage[self.positional_info_position_to_index[pos] + i + 1] += 1

                    if length_of_insert > total_insert_bases:
                        dot_bases = length_of_insert - total_insert_bases
                        for i in range(dot_bases):
                            base = '*'
                            base_q = MIN_DELETE_QUALITY
                            cigar_code = 2
                            ref_base = ''
                            pileup_attributes = (base, base_q, mapping_quality, cigar_code, strand_direction)
                            channel_object = ImageChannels(pileup_attributes, ref_base, 0)
                            read_to_image_row.append(channel_object.get_channels_except_support())

                            indx = self.positional_info_position_to_index[pos] + total_insert_bases + i + 1
                            self.base_frequency[indx][base] += 1
                            self.index_based_coverage[indx] += 1

            self.image_row_for_reads[read_id] = (read_to_image_row, star_pos, end_pos)

    def postprocess_reference(self):
        left_position = min(self.reference_dictionary.keys())
        right_position = max(self.reference_dictionary.keys()) + 1
        reference_to_image_row = []
        index = 0

        for pos in range(left_position, right_position):
            base = self.reference_dictionary[pos] if pos in self.reference_dictionary else 'N'
            reference_to_image_row.append(ImageChannels.get_channels_for_ref(base))
            self.positional_info_index_to_position[index] = (pos, False)
            self.positional_info_position_to_index[pos] = index
            self.reference_base_by_index[index] = base
            index += 1
            if pos in self.insert_length_info:
                for i in range(self.insert_length_info[pos]):
                    base = '*'
                    reference_to_image_row.append(ImageChannels.get_channels_for_ref(base))
                    self.positional_info_index_to_position[index] = (pos, True)
                    self.reference_base_by_index[index] = base
                    index += 1

        self.image_row_for_ref = (reference_to_image_row, left_position, right_position)

    def get_reference_row(self, start_pos, end_pos):
        ref_row, ref_start, ref_end = self.image_row_for_ref
        start_index = self.positional_info_position_to_index[start_pos] - self.positional_info_position_to_index[ref_start]
        end_index = self.positional_info_position_to_index[end_pos] - self.positional_info_position_to_index[ref_start]
        ref_row = np.array(ref_row[start_index:end_index])

        return ref_row

    def get_read_row(self, read_id, read_info, image_start, image_end, image_width):
        read_start, read_end, mq, is_rev = read_info
        read_row = self.image_row_for_reads[read_id][0]

        read_start_new = read_start
        read_end_new = read_end
        if image_start > read_start:
            read_start_new = image_start

        if image_end < read_end:
            read_end_new = image_end

        start_index = self.positional_info_position_to_index[read_start_new] - \
                      self.positional_info_position_to_index[read_start]
        end_index = self.positional_info_position_to_index[read_end_new] - \
                      self.positional_info_position_to_index[read_start]

        image_row = read_row[start_index:end_index]

        if image_start < read_start_new:
            distance = self.positional_info_position_to_index[read_start_new] - \
                      self.positional_info_position_to_index[image_start]
            empty_channels_list = [ImageChannels.get_empty_channels()] * int(distance)
            image_row = empty_channels_list + image_row

        return image_row, read_start_new, read_end_new

    @staticmethod
    def process_rows(image, image_width):
        for i, image_row in enumerate(image):
            if len(image_row) < image_width:
                length_difference = image_width - len(image_row)
                empty_channels = [ImageChannels.get_empty_channels() for i in range(length_difference)]
                image[i] = image[i] + empty_channels
            elif len(image_row) > image_width:
                image[i] = image[i][:image_width]

        return image

    @staticmethod
    def get_row_for_read(read_start, read_end, row_info, image_height):
        for i in range(image_height):
            if read_start > row_info[i]:
                return i

        return -1

    def create_image(self, interval_start, interval_end, read_id_list, image_height=300):
        whole_image = [[] for i in range(image_height)]
        image_row_info = defaultdict(int)

        # get all reads that align to that position
        # O(n)
        ref_row = self.get_reference_row(interval_start, interval_end)
        image_width = ref_row.shape[0]
        whole_image[0].extend(ref_row)
        image_row_info[0] = interval_end

        # O(n)
        for read_id in read_id_list:
            read_start, read_end, mq, is_rev = self.read_info[read_id]
            row = self.get_row_for_read(read_start, read_end, image_row_info, image_height)
            if row < 0:
                continue

            image_start = max(image_row_info[row], interval_start)

            read_row, read_start, read_end = self.get_read_row(read_id, self.read_info[read_id], image_start,
                                                               interval_end, image_width)
            whole_image[row].extend(read_row)
            image_row_info[row] = read_end
            # print('After update', read_start, read_end, row, image_row_info[row])

        whole_image = self.process_rows(whole_image, image_width)

        whole_image = np.array(whole_image)

        # analyze_np_array(whole_image, image_width, image_height)
        return whole_image

    def process_interval(self, interval_start, interval_end):
        """
        Processes all reads and reference for this interval
        :param interval_start:
        :param interval_end:
        :return:
        """
        # get all the reads of that region
        reads = self.bam_handler.get_reads(self.chromosome_name, interval_start, interval_end)

        total_reads = 0
        read_id_list = []
        for read in reads:
            # check if the read is usable
            if read.mapping_quality >= DEFAULT_MIN_MAP_QUALITY and read.is_secondary is False \
                    and read.is_supplementary is False and read.is_unmapped is False and read.is_qcfail is False:
                # for paired end make sure read name is unique
                read.query_name = read.query_name + '_1' if read.is_read1 else read.query_name + '_2'
                self.process_read(read, interval_start, interval_end)
                read_id_list.append(read.query_name)
                total_reads += 1

        self.postprocess_reference()
        self.postprocess_reads(read_id_list)

        regional_pileup = self.create_image(interval_start, interval_end, read_id_list)

        if DEBUG_MESSAGE:
            sys.stderr.write(TextColor.BLUE)
            sys.stderr.write("INFO: TOTAL READ IN REGION: " + self.chromosome_name + " " + str(interval_start) + " " +
                             str(interval_end) + " " + str(total_reads) + "\n" + TextColor.END)

        return regional_pileup

    @staticmethod
    def save_image_as_png(pileup_array, save_dir, file_name):
        pileup_array_2d = pileup_array.reshape((pileup_array.shape[0], -1))
        try:
            misc.imsave(save_dir + file_name + ".png", pileup_array_2d, format="PNG")
        except:
            sys.stderr.write(TextColor.RED)
            sys.stderr.write("ERROR: ERROR SAVING FILE: " + save_dir + file_name + ".png" + "\n" + TextColor.END)
            sys.stderr.write()

    def get_allele_bases_from_vcf_genotype(self, indx, vcf_records, base_frequencies):
        bases = []
        for vcf_record in vcf_records:
            allele, genotype = vcf_record
            if not base_frequencies[allele]:
                warn_msg = str(indx) + " " + str(vcf_record) + "\n"
                sys.stderr.write(WARN_COLOR + " WARN: VCF ALLELE NOT IN BAM: " + warn_msg + TextColor.END)
            # hom_alt
            if genotype[0] == genotype[1]:
                bases.append(allele)
                bases.append(allele)
            else:
                bases.append(allele)
        if len(bases) == 0:
            return '-', '-'
        elif len(bases) == 1:
            return bases[0], '-'
        else:
            return bases[0], bases[1]

    def get_estimated_allele_strings(self, interval_start, interval_end):
        start_index = self.positional_info_position_to_index[interval_start]
        end_index = self.positional_info_position_to_index[interval_end]
        string_a = ''
        string_b = ''
        vcf_string_a = ''
        vcf_string_b = ''
        positional_values = ''
        for i in range(start_index, end_index):
            pos_increase = 0 if self.positional_info_index_to_position[i][1] is True else 1
            positional_values = positional_values + str(pos_increase)
            vcf_alts = []
            if i in self.vcf_positional_dict:
                alt_a, alt_b = self.get_allele_bases_from_vcf_genotype(i, self.vcf_positional_dict[i],
                                                                       self.base_frequency[i])
                vcf_string_a += alt_a
                vcf_string_b += alt_b
                vcf_alts.append(alt_a)
                vcf_alts.append(alt_b)
            else:
                vcf_string_a += '-'
                vcf_string_b += '-'
            total_bases = self.index_based_coverage[i]
            bases = []
            ref_base = self.reference_base_by_index[i]

            for base in self.base_frequency[i]:
                base_frequency = self.base_frequency[i][base] / total_bases
                if base_frequency >= ALLELE_FREQUENCY_THRESHOLD_FOR_REPORTING:
                    bases.append(base)
                    # warn if a really abundant allele is not in VCF
                    if base not in vcf_alts and base != ref_base:
                        genome_position = self.positional_info_index_to_position[i][0]
                        msg = str(genome_position) + " " + str(ref_base) + " " + str(base) + " " + \
                              str(int(base_frequency*100))
                        sys.stderr.write(WARN_COLOR + ' WARN: BAM ALLELE NOT FOUND IN VCF ' + msg + "%\n" + TextColor.END)
            if len(bases) == 0:
                string_a += '-'
                string_b += '-'
            elif len(bases) == 1:
                string_a += bases[0] if bases[0] != ref_base else '-'
                string_b += bases[0] if bases[0] != ref_base else '-'
            else:
                string_a += bases[0] if bases[0] != ref_base else '-'
                string_b += bases[1] if bases[1] != ref_base else '-'
        return string_a, string_b, vcf_string_a, vcf_string_b, positional_values

    def populate_vcf_alleles(self, positional_vcf, interval_start, interval_end):
        for pos in range(interval_start, interval_end):
            bam_pos = pos + VCF_INDEX_BUFFER
            indx = self.positional_info_position_to_index[bam_pos]

            if pos in positional_vcf.keys():
                snp_recs, in_recs, del_recs = positional_vcf[pos]
                alt_alleles_found = self.positional_allele_dictionary[bam_pos]

                for snp_rec in snp_recs:
                    alt_ = (snp_rec.alt, 1)
                    if alt_ in alt_alleles_found:
                        self.vcf_positional_dict[indx].append((snp_rec.alt, snp_rec.genotype))
                    else:
                        sys.stderr.write(WARN_COLOR + "WARN: VCF RECORD ALLELE NOT FOUND: " + str(snp_rec) + "\n" + TextColor.END)

                for in_rec in in_recs:
                    alt_ = (in_rec.alt, 2)
                    if alt_ in alt_alleles_found:
                        for i in range(1, len(in_rec.alt)):
                            self.vcf_positional_dict[indx+i].append((in_rec.alt[i], in_rec.genotype))
                    else:
                        sys.stderr.write(WARN_COLOR + "WARN: VCF RECORD ALLELE NOT FOUND: " + str(in_rec) + "\n" + TextColor.END)

                for del_rec in del_recs:
                    alt_ = (del_rec.ref, 3)
                    if alt_ in alt_alleles_found:
                        for i in range(1, len(del_rec.ref)):
                            del_indx = self.positional_info_position_to_index[bam_pos+i]
                            self.vcf_positional_dict[del_indx].append(('.', del_rec.genotype))
                    else:
                        sys.stderr.write(WARN_COLOR + "WARN: VCF RECORD ALLELE NOT FOUND: " + str(del_rec) + "\n" + TextColor.END)

    def create_region_alignment_image(self, interval_start, interval_end, positional_variants):
        """
        Generate labeled images of a given region of the genome
        :param interval_start: Starting genomic position of the interval
        :param interval_end: End genomic position of the interval
        :param positional_variants: List of positional variants in that region
        :return:
        """
        image = self.process_interval(interval_start, interval_end)
        self.populate_vcf_alleles(positional_variants, interval_start, interval_end)
        alt_a, alt_b, vcf_a, vcf_b, pos_vals= self.get_estimated_allele_strings(interval_start, interval_end)

        return image, alt_a, alt_b, vcf_a, vcf_b, pos_vals
