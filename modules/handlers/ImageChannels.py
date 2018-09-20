MAX_COLOR_VALUE = 254
BASE_QUALITY_CAP = 40
MAP_QUALITY_CAP = 60
MAP_QUALITY_FILTER = 5
MIN_DELETE_QUALITY = 0
IMAGE_DEPTH_THRESHOLD = 300

global_base_color_dictionary = {'A': 250, 'C': 30, 'G': 180, 'T': 100, '*': 10, '.': 5, 'N': 10}


class imageChannels:
    """
    Handles how many channels to create for each base and their way of construction.
    """

    def __init__(self, base, base_qual, map_qual, is_rev, ref_base):
        """
        Initialize a base with it's attributes
        :param pileup_attributes: Attributes of a pileup base
        :param ref_base: Reference base corresponding to that pileup base
        """
        self.pileup_base = base
        self.base_qual = base_qual
        self.map_qual = map_qual
        self.is_rev = is_rev
        self.ref_base = ref_base
        self.is_match = True if self.ref_base == self.pileup_base else False

    @staticmethod
    def get_empty_channels():
        """
        Get empty channel values
        :return:
        """
        return [0, 0, 0, 0, 0, 0]

    def get_channels_except_support(self):
        """
                Get a bases's channel construction
                :return: [color spectrum of channels based on base attributes]
        """
        if self.pileup_base == '.':
            return [0, 0, 0, 0, 0, 0]

        base_color = global_base_color_dictionary[self.pileup_base] \
            if self.pileup_base in global_base_color_dictionary else 0
        base_quality_color = int(MAX_COLOR_VALUE * (min(self.base_qual, BASE_QUALITY_CAP) / BASE_QUALITY_CAP))
        map_quality_color = int(MAX_COLOR_VALUE * (min(self.map_qual, MAP_QUALITY_CAP) / MAP_QUALITY_CAP))
        strand_color = 240 if self.is_rev else 70
        match_color = int(MAX_COLOR_VALUE * 0.2) if self.is_match is True else int(MAX_COLOR_VALUE * 1.0)
        # assume all the reads is not supporting as that's the majority of cases
        support_color = 1

        return [base_color, base_quality_color, map_quality_color, strand_color, match_color, support_color]

    @staticmethod
    def get_channels_for_ref(base):
        """
        Get a reference bases's channel construction
        :param base: Reference base
        :return: [color spectrum of channels based on some default values]
        """
        # base color
        base_color = global_base_color_dictionary[base] if base in global_base_color_dictionary else 0
        # highest base quality
        base_quality_color = int(MAX_COLOR_VALUE)
        # highest map quality
        map_quality_color = int(MAX_COLOR_VALUE)
        # strand is forward
        strand_color = 240
        # matches ref
        match_color = int(MAX_COLOR_VALUE * 0.2)
        # not supporing alt
        support_color = int(MAX_COLOR_VALUE * 0.6)
        return [base_color, base_quality_color, map_quality_color, strand_color, match_color, support_color]
