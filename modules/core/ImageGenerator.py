import sys
import numpy as np
from scipy import misc
import h5py
import os
from operator import itemgetter
from modules.handlers.ImageChannels import imageChannels
from modules.core.CandidateFinder import MATCH_ALLELE, INSERT_ALLELE, DELETE_ALLELE
from modules.core.CandidateLabeler import CandidateLabeler
from modules.handlers.TextColor import TextColor

"""
This script creates pileup images given a vcf record, bam alignment file and reference fasta file.

imageChannels: Handles how many channels to create for each base and their structure

"""
MAP_QUALITY_FILTER = 5
MAX_COLOR_VALUE = 254
BASE_QUALITY_CAP = 40
MAP_QUALITY_CAP = 60
MIN_DELETE_QUALITY = 20
MATCH_CIGAR_CODE = 0
INSERT_CIGAR_CODE = 1
DELETE_CIGAR_CODE = 2

HOM_CLASS = 0
HET_CLASS = 1
HOM_ALT_CLASS = 2


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
    def get_class_label_for_alt1(gt):
        h1, h2 = gt
        if h1 == 1 or h2 == 1:
            if h1 == h2:
                return HOM_ALT_CLASS
            else:
                return HET_CLASS
        return HOM_CLASS

    @staticmethod
    def get_class_label_for_alt2(gt):
        h1, h2 = gt
        if h1 == 2 or h2 == 2:
            if h1 == h2:
                return HOM_ALT_CLASS
            else:
                return HET_CLASS
        return HOM_CLASS

    @staticmethod
    def get_class_label_for_combined_alt(gt):
        h1, h2 = gt
        if h1 == 0 and h2 == 0:
            return HOM_CLASS

        if h1 == 0 or h2 == 0:
            return HET_CLASS

        return HOM_ALT_CLASS

    @staticmethod
    def get_combined_records_for_two_alts(record):
        """
        Returns records for sites where we have two alternate alleles.
        :param record: Record that belong to the site
        :return: Records of a site
        """
        chr_name, pos_start, pos_end, ref, alt1, alt2, rec_type_alt1, rec_type_alt2 = record[0:8]
        # get the genotypes from the record
        gt = record[-1]
        gt1 = ImageGenerator.get_class_label_for_alt1(gt)
        gt2 = ImageGenerator.get_class_label_for_alt2(gt)
        # get the genotype of the images where both of these alleles are used together
        gt3 = ImageGenerator.get_class_label_for_combined_alt(gt)

        # create two separate records for each of the alleles
        rec_1 = [chr_name, pos_start, pos_end, ref, alt1, '.', rec_type_alt1, 0, gt1]
        rec_2 = [chr_name, pos_start, pos_end, ref, alt2, '.', rec_type_alt2, 0, gt2]
        if gt3 is not None:
            # if gt3 is not invalid create the record where both of the alleles are used together
            rec_3 = [chr_name, pos_start, pos_end, ref, alt1, alt2, rec_type_alt1, rec_type_alt2, gt3]
            return [rec_1, rec_2, rec_3]

        return [rec_1, rec_2]

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

        # huge insert both on left and right
        if left_pos == right_pos:
            right_pos += 1

        left_padding = 0
        if left_length < half_of_width:
            left_padding = int(half_of_width) - left_length

        return left_pos, right_pos, left_padding

    def get_reference_row(self, start_pos, end_pos, left_pad, image_width):
        ref_row, ref_start, ref_end = self.image_row_for_ref
        start_index = self.positional_info_position_to_index[start_pos] - self.positional_info_position_to_index[ref_start]
        end_index = self.positional_info_position_to_index[end_pos] - self.positional_info_position_to_index[ref_start]
        ref_row = np.array(ref_row[start_index:end_index], dtype=np.uint8)

        if left_pad > 0:
            empty_channels_list = [imageChannels.get_empty_channels()] * int(left_pad)
            ref_row = np.concatenate((np.array(empty_channels_list, dtype=np.uint8), ref_row), axis=0)

        if len(ref_row) < image_width:
            empty_channels_list = [imageChannels.get_empty_channels()] * int(image_width - len(ref_row))
            ref_row = np.concatenate((ref_row, np.array(empty_channels_list, dtype=np.uint8)), axis=0)

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
        support_val = 254 if supporting is True else 152

        read_row = np.array(self.image_row_for_reads[read_id][0], dtype=np.uint8)
        read_row[:, 5] = read_row[:, 5] * support_val

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
            # replacing this with native list operation may make the code faster
            image_row = np.concatenate((np.array(empty_channels_list, dtype=np.uint8), image_row), axis=0)

        if left_pad:
            empty_channels_list = [imageChannels.get_empty_channels()] * int(left_pad)
            # replacing this with native list operation may make the code faster
            image_row = np.concatenate((np.array(empty_channels_list, dtype=np.uint8), image_row), axis=0)

        if len(image_row) < image_width:
            empty_channels_list = [imageChannels.get_empty_channels()] * int(image_width - len(image_row))
            # replacing this with native list operation may make the code faster
            image_row = np.concatenate((image_row, np.array(empty_channels_list, dtype=np.uint8)), axis=0)
        if len(image_row) > image_width:
            image_row = image_row[:image_width]

        return image_row

    @staticmethod
    def add_empty_rows(image, empty_rows_to_add, image_width):
        for i in range(empty_rows_to_add):
            empty_channels = [imageChannels.get_empty_channels()] * image_width
            image.append(empty_channels)
        return image

    def create_image(self, query_start_pos, query_end_pos, ref, alts, alt_types,
                     image_height, image_width, ref_band=5):
        alts_norm = []
        for i, alt in enumerate(alts):
            alts_norm.append((alt, alt_types[i]))
        alts = alts_norm
        whole_image = []
        # get all reads that align to that position
        # O(n)
        left_pos, right_pos = self.get_left_right_genomic_position(query_start_pos, image_width)
        # O(n)
        start_pos, end_pos, left_pad = self.get_start_end_based_on_image_width(query_start_pos, image_width, left_pos, right_pos)

        ref_row = self.get_reference_row(start_pos, end_pos, left_pad, image_width)

        for i in range(ref_band):
            whole_image.append(ref_row)

        reads = self.positional_read_info[query_start_pos]
        # is_intersectable = True
        # for alt in alts:
        #     if alt[1] != DELETE_ALLELE:
        #         is_intersectable = False
        #
        # if is_intersectable is True:
        #     for pos in range(query_start_pos+1, query_end_pos+1):
        #         intersected_reads = set(intersected_reads).intersection(self.positional_read_info[pos])

        reads = sorted(reads, key=itemgetter(1))
        # O(n)
        for read in reads:
            read_row = self.get_read_row(query_start_pos, alts, read, start_pos, end_pos, left_pad, image_width)
            if len(whole_image) < image_height:
                whole_image.append(read_row)
            else:
                break

        for i in range(image_height - len(whole_image)):
            empty_channels = [[0, 0, 0, 0, 0, 0]] * image_width
            whole_image.append(empty_channels)

        return np.array(whole_image, dtype=np.uint8)

    @staticmethod
    def generate_and_save_candidate_images(chromosome_name, candidate_list, image_generator, thread_no, output_dir,
                                           image_height, image_width, image_channels=6):
        """
        Generate and save images from a given labeled candidate list.
        :param chromosome_name: Name of the chromosome which is being processed
        :param candidate_list: List of candidates.
        :param image_generator: Image generator object containing all dictionaries to generate the images.
        :param thread_no: The thread number used to name the files.
        :param output_dir: where to save all the records
        :param image_height: Height of the image (Coverage)
        :param image_width: Width of the image (Context)
        :param image_channels: Number of feature channels for the image
        :return:
        """
        # declare the size of the image
        if len(candidate_list) == 0:
            return

        # create summary file where the location of each image is recorded
        contig = str(chromosome_name)
        smry = open(output_dir + "summary/" + "summary" + '_' + contig + "_" + str(thread_no) + ".csv", 'w')
        # create a h5py file where the images are stored
        hdf5_filename = output_dir + contig + '_' + str(thread_no) + ".h5"
        hdf5_file = h5py.File(hdf5_filename, mode='w')
        # list of image records to be generated
        image_record_set = []
        # expand the records for sites where two alleles are found
        for record in candidate_list:
            chr_name, pos_start, pos_end, ref, alt1, alt2, rec_type_alt1, rec_type_alt2 = record[0:8]
            gt = record[-1]

            if alt2 != '.':
                image_record_set.extend(ImageGenerator.get_combined_records_for_two_alts(record))
            else:
                gt1 = ImageGenerator.get_class_label_for_alt1(gt)
                image_record_set.append([chr_name, pos_start, pos_end, ref, alt1, alt2, rec_type_alt1, rec_type_alt2,
                                         gt1])

        # set of images and labels we are generating
        img_set = []
        label_set = []
        # index of the image we generate the images
        indx = 0
        for img_record in image_record_set:
            chr_name, pos_start, pos_end, ref, alt1, alt2, rec_type_alt1, rec_type_alt2, label = img_record

            # list of alts in this record
            alts = [alt1]
            # list of type of record (IN, DEL, SNP)
            rec_types = [rec_type_alt1]
            if alt2 != '.':
                alts.append(alt2)
                rec_types.append(rec_type_alt2)
            # the image array
            image_array = image_generator.create_image(pos_start, pos_end, ref, alts, rec_types,
                                                       image_height=image_height, image_width=image_width)

            # the record of the image we want to save in the summary file
            img_rec = str('\t'.join(str(item) for item in img_record))
            label_set.append(label)
            img_set.append(np.array(image_array, dtype=np.uint8))
            smry.write(os.path.abspath(hdf5_filename) + ',' + str(indx) + ',' + img_rec + '\n')
            indx += 1

        # the image dataset we save. The index name in h5py is "images".
        img_dataset = hdf5_file.create_dataset("images", (len(img_set),) + (image_height, image_width, image_channels),
                                               np.uint8, compression='gzip')
        # the labels for images that we saved
        label_dataset = hdf5_file.create_dataset("labels", (len(label_set),), np.uint8)
        # save the images and labels to the h5py file
        img_dataset[...] = img_set
        label_dataset[...] = label_set
