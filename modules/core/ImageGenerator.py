import sys
from scipy import misc
from modules.handlers.ImageChannels import imageChannels
from modules.core.ImageAnalyzer import *

"""
This script creates pileup images given a vcf record, bam alignment file and reference fasta file.

imageChannels: Handles how many channels to create for each base and their structure

"""
MAP_QUALITY_FILTER = 5.0
MAX_COLOR_VALUE = 254.0
BASE_QUALITY_CAP = 40.0
MAP_QUALITY_CAP = 60.0
MIN_DELETE_QUALITY = 20.0
MATCH_CIGAR_CODE = 0
INSERT_CIGAR_CODE = 1
DELETE_CIGAR_CODE = 2
IMAGE_DEPTH_THRESHOLD = 300


class ImageGenerator:
    """
    Processes a pileup around a position
    """
    def __init__(self, dictionaries):
        self.image_row_for_reads = dictionaries[0]
        self.image_row_for_ref = dictionaries[1]
        self.positional_info_position_to_index = dictionaries[2]
        self.positional_info_index_to_position = dictionaries[3]
        self.allele_dictionary = dictionaries[4]
        self.positional_read_info = dictionaries[5]

    @staticmethod
    def save_image_as_png(pileup_array, save_dir, file_name):
        pileup_array_2d = pileup_array.reshape((pileup_array.shape[0], -1))
        try:
            misc.imsave(save_dir + file_name + ".png", pileup_array_2d, format="PNG")
        except:
            sys.stderr.write("ERROR SAVING FILE: " + save_dir + file_name + ".png" + "\n")

    def get_left_right_genomic_position(self, pos, image_width):
        left_genomic_position = self.positional_read_info[pos][0][1]
        right_genomic_position = self.positional_read_info[pos][0][2]
        for read in self.positional_read_info[pos]:
            left_genomic_position = min(left_genomic_position, read[1])
            right_genomic_position = max(right_genomic_position, read[2])
        return left_genomic_position, right_genomic_position

    def get_start_end_based_on_image_width(self, pos, image_width, left_pos, right_pos):
        pos_index = self.positional_info_position_to_index[pos]
        left_index = self.positional_info_position_to_index[left_pos]
        right_index = self.positional_info_position_to_index[right_pos]
        left_length = pos_index - left_index
        half_of_width = int(image_width / 2)

        if left_length > half_of_width:
            while left_length > half_of_width:
                left_pos += 1
                left_index = self.positional_info_position_to_index[left_pos]
                left_length = pos_index - left_index

        right_length = right_index - pos_index
        if right_length > half_of_width:
            while right_length > half_of_width:
                right_pos -= 1
                right_index = self.positional_info_position_to_index[right_pos]
                right_length = right_index - pos_index

        left_padding = 0
        if left_length < half_of_width:
            left_padding = int(half_of_width) - left_length

        return left_pos, right_pos, left_padding

    def get_reference_row(self, start_pos, end_pos, left_pad, image_width):
        ref_row, ref_start, ref_end = self.image_row_for_ref
        start_index = self.positional_info_position_to_index[start_pos] - self.positional_info_position_to_index[ref_start]
        end_index = self.positional_info_position_to_index[end_pos] - self.positional_info_position_to_index[ref_start]
        ref_row = np.array(ref_row[start_index:end_index])
        if left_pad > 0:
            empty_channels_list = [imageChannels.get_empty_channels()] * int(left_pad)
            ref_row = np.concatenate((np.array(empty_channels_list), ref_row), axis=0)
        if len(ref_row) < image_width:
            empty_channels_list = [imageChannels.get_empty_channels()] * int(image_width - len(ref_row))
            ref_row = np.concatenate((ref_row, np.array(empty_channels_list)), axis=0)
        if len(ref_row) > image_width:
            ref_row = ref_row[:image_width]

        return ref_row

    def get_distance_between_two_positions(self, pos_a, pos_b):
        left_index = self.positional_info_position_to_index[pos_a]
        right_index = self.positional_info_position_to_index[pos_b]
        return right_index - left_index

    def get_read_row(self, pos, alts, read, image_start, image_end, left_pad, image_width):
        read_id, read_start, read_end = read
        supporting = False
        if read_id in self.allele_dictionary and pos in self.allele_dictionary[read_id]:
            read_alleles = self.allele_dictionary[read_id][pos]
            for alt in alts:
                if alt in read_alleles:
                    supporting = True
                    break
        support_val = 254.0 if supporting is True else 152.4

        read_row = np.array(self.image_row_for_reads[read_id][0])
        read_row = np.insert(read_row, 6, support_val, axis=1)

        read_start_new = read_start
        read_end_new = read_end
        if image_start > read_start:
            read_start_new = image_start

        if image_end < read_end:
            read_end_new = image_end
        # print(image_start, image_end, read_start, read_end, read_start_new, read_end_new)

        start_index = self.positional_info_position_to_index[read_start_new] - \
                      self.positional_info_position_to_index[read_start]
        end_index = self.positional_info_position_to_index[read_end_new] - \
                      self.positional_info_position_to_index[read_start]
        # print(start_index, end_index, end_index-start_index)
        image_row = read_row[start_index:end_index]

        if image_start < read_start_new:
            distance = self.positional_info_position_to_index[read_start_new] - \
                      self.positional_info_position_to_index[image_start]
            empty_channels_list = [imageChannels.get_empty_channels()] * int(distance)
            image_row = np.concatenate((np.array(empty_channels_list), image_row), axis=0)

        if left_pad:
            empty_channels_list = [imageChannels.get_empty_channels()] * int(left_pad)
            image_row = np.concatenate((np.array(empty_channels_list), image_row), axis=0)

        if len(image_row) < image_width:
            empty_channels_list = [imageChannels.get_empty_channels()] * int(image_width - len(image_row))
            image_row = np.concatenate((image_row, np.array(empty_channels_list)), axis=0)
        if len(image_row) > image_width:
            image_row = image_row[:image_width]

        return image_row

    @staticmethod
    def add_empty_rows(image, empty_rows_to_add, image_width):
        for i in range(empty_rows_to_add):
            empty_channels = [imageChannels.get_empty_channels()] * image_width
            image.append(empty_channels)
        return image

    def create_image(self, query_pos, ref, alts, alt_types, image_height=300, image_width=300, ref_band=5):
        alts_norm = []
        for i, alt in enumerate(alts):
            alts_norm.append((alt, alt_types[i]))
        alts = alts_norm
        whole_image = []
        # get all reads that align to that position
        # O(n)
        left_pos, right_pos = self.get_left_right_genomic_position(query_pos, image_width)
        # O(n)
        start_pos, end_pos, left_pad = self.get_start_end_based_on_image_width(query_pos, image_width, left_pos, right_pos)

        ref_row = self.get_reference_row(start_pos, end_pos, left_pad, image_width)

        for i in range(ref_band):
            whole_image.append(ref_row)

        # O(n)
        for read in self.positional_read_info[query_pos]:
            read_row = self.get_read_row(query_pos, alts, read, start_pos, end_pos, left_pad, image_width)
            if len(whole_image) < image_height:
                whole_image.append(read_row)
            else:
                break

        whole_image = self.add_empty_rows(whole_image, image_height - len(whole_image), image_width)
        whole_image = np.array(whole_image)

        # analyze_np_array(whole_image, image_width, image_height)
        return whole_image
