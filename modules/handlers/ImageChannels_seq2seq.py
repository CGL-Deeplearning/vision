MAX_COLOR_VALUE = 254.0
BASE_QUALITY_CAP = 40.0
MAP_QUALITY_CAP = 60.0
MAP_QUALITY_FILTER = 5.0
MIN_DELETE_QUALITY = 20.0
MATCH_CIGAR_CODE = 0
INSERT_CIGAR_CODE = 1
DELETE_CIGAR_CODE = 2
IMAGE_DEPTH_THRESHOLD = 300

global_base_color_dictionary = {'A': 254.0, 'C': 100.0, 'G': 180.0, 'T': 30.0, '*': 5.0, '.': 5.0, 'N': 5.0}
global_cigar_color_dictionary = {0: MAX_COLOR_VALUE, 1: MAX_COLOR_VALUE*0.6, 2: MAX_COLOR_VALUE*0.3}


class ImageChannels:
    """
    Handles how many channels to create for each base and their way of construction.
    """

    def __init__(self, pileup_attributes, ref_base, is_supporting):
        """
        Initialize a base with it's attributes
        :param pileup_attributes: Attributes of a pileup base
        :param ref_base: Reference base corresponding to that pileup base
        """
        self.pileup_base = pileup_attributes[0]
        self.base_qual = pileup_attributes[1]
        self.map_qual = pileup_attributes[2]
        self.cigar_code = pileup_attributes[3]
        self.is_rev = pileup_attributes[4]
        self.ref_base = ref_base
        self.is_match = True if self.ref_base == self.pileup_base else False

    @staticmethod
    def get_base_color(base):
        """
        Get color based on a base.
        - Uses different band of the same channel.
        :param base:
        :return:
        """
        if base == 'A':
            return 254.0
        if base == 'C':
            return 100.0
        if base == 'G':
            return 180.0
        if base == 'T':
            return 30.0
        if base == '*' or 'N':
            return 5.0

    @staticmethod
    def get_base_quality_color(base_quality):
        """
        Get a color spectrum given base quality
        :param base_quality: value of base quality
        :return:
        """
        c_q = min(base_quality, BASE_QUALITY_CAP)
        color = (MAX_COLOR_VALUE * c_q) / BASE_QUALITY_CAP
        return color

    @staticmethod
    def get_map_quality_color(map_quality):
        """
        Get a color spectrum given mapping quality
        :param map_quality: value of mapping quality
        :return:
        """
        c_q = min(map_quality, MAP_QUALITY_CAP)
        color = (MAX_COLOR_VALUE * c_q) / MAP_QUALITY_CAP
        return color

    @staticmethod
    def get_strand_color(is_rev):
        """
        Get color for forward and reverse reads
        :param is_rev: True if read is reversed
        :return:
        """
        if is_rev is True:
            return 240.0
        else:
            return 70.0

    @staticmethod
    def get_match_ref_color(is_match):
        """
        Get color for base matching to reference
        :param is_match: If true, base matches to reference
        :return:
        """
        if is_match is True:
            return MAX_COLOR_VALUE * 0.2
        else:
            return MAX_COLOR_VALUE * 1.0

    @staticmethod
    def get_cigar_color(cigar_code):
        """
        ***NOT USED YET***
        :param is_in_support:
        :return:
        """
        if cigar_code == 0:
            return MAX_COLOR_VALUE
        if cigar_code == 1:
            return MAX_COLOR_VALUE * 0.6
        if cigar_code == 2:
            return MAX_COLOR_VALUE * 0.3

    @staticmethod
    def get_empty_channels():
        """
        Get empty channel values
        :return:
        """
        return [0, 0, 0, 0, 0, 0]

    def get_channels(self):
        """
        Get a bases's channel construction
        :return: [color spectrum of channels based on base attributes]
        """
        base_color = global_base_color_dictionary[self.pileup_base] \
            if self.pileup_base in global_base_color_dictionary else 0.0
        base_quality_color = (MAX_COLOR_VALUE * min(self.base_qual, BASE_QUALITY_CAP)) / BASE_QUALITY_CAP
        map_quality_color = (MAX_COLOR_VALUE * min(self.map_qual, MAP_QUALITY_CAP)) / MAP_QUALITY_CAP
        strand_color = 240.0 if self.is_rev else 70.0
        match_color = MAX_COLOR_VALUE * 0.2 if self.is_match is True else MAX_COLOR_VALUE * 1.0
        cigar_color = global_cigar_color_dictionary[self.cigar_code] \
            if self.cigar_code in global_cigar_color_dictionary else 0.0
        return [base_color, base_quality_color, map_quality_color, strand_color, match_color, cigar_color]

    # def get_channels_except_support(self):
    #     """
    #     Get a bases's channel construction
    #     :return: [color spectrum of channels based on base attributes]
    #     """
    #     base_color = self.get_base_color(self.pileup_base)
    #     base_quality_color = self.get_base_quality_color(self.base_qual)
    #     map_quality_color = self.get_map_quality_color(self.map_qual)
    #     strand_color = self.get_strand_color(self.is_rev)
    #     match_color = self.get_match_ref_color(self.is_match)
    #     cigar_color = self.get_cigar_color(self.cigar_code)
    #     return [base_color, base_quality_color, map_quality_color, strand_color, match_color, cigar_color]

    @staticmethod
    def get_channels_for_ref(base):
        """
        Get a reference bases's channel construction
        :param base: Reference base
        :return: [color spectrum of channels based on some default values]
        """
        cigar_code = MATCH_CIGAR_CODE if base != '*' else INSERT_CIGAR_CODE
        base_color = ImageChannels.get_base_color(base)
        base_quality_color = ImageChannels.get_base_quality_color(BASE_QUALITY_CAP)
        map_quality_color = ImageChannels.get_map_quality_color(60)
        strand_color = ImageChannels.get_strand_color(is_rev=False)
        match_color = ImageChannels.get_match_ref_color(is_match=True)
        cigar_color = ImageChannels.get_cigar_color(cigar_code)
        return [base_color, base_quality_color, map_quality_color, strand_color, match_color, cigar_color]
