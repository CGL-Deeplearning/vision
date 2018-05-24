import pandas

"""
            possible combos:
            gt1       gt2     Candidate validated?
            --------------------------------------
            hom       hom     no
            het       hom     yes
            het       het     yes
            hom_alt   hom     yes
            hom       None    no
            het       None    yes
            hom_alt   None    yes

            impossible combos:
            gt1       gt2     Candidate validated?
            --------------------------------------
            hom_alt   hom_alt NA
            het       hom_alt NA
            hom       hom_alt NA
            hom_alt   het     NA
            hom       het     NA
            None      None    NA

"""
GENOTYPE_DICT = {"Hom": 0, "Het": 1, "Hom_alt": 2}
GENOTYPE_NAMES = ["Hom", "Het", "Hom_alt"]
VCF_OFFSET = 1
PLOIDY = 4

QUALITY_SCALING_CONSTANT = 1000

# DEBUG_FREQUENCIES = False
DEBUG_PRINT_ALL = False

# Candidate data indexes
CHR_NAME = 0
START = 1
STOP = 2
REF_INDEX = 3
ALT1 = 4
ALT2 = 5
ALT1_TYPE = 6
ALT2_TYPE = 7

FREQUENCIES = -6
COVERAGE = -5
MAP_QUALITY_NON_REF = -4
BASE_QUALITY_NON_REF = -3
MAP_QUALITY_REF = -2
BASE_QUALITY_REF = -1

# Positional vcf indexes
SNP, IN, DEL = 0, 1, 2
SNP_CANDIDATE, IN_CANDIDATE, DEL_CANDIDATE = 1, 2, 3
SNP_DEL = 3

# VCF record indexes
REF, ALT, GT = 0, 1, 2

# Genotype codes
HOM, HET, HOM_ALT = 0, 1, 2

class CandidateVectorizer:
    def __init__(self, fasta_handler):
        """
        Initialize candidateLabeler object
        :param fasta_handler: module that fetches reference sequence substrings from a FASTA file
        """
        self.fasta_handler = fasta_handler      # unfortunately need to query reference sequence on a per-site basis

    def _generate_fixed_size_freq_list(self, site_frequencies, vector_length=PLOIDY):
        frequency_list = list()

        # concatenate frequencies for Match, Insert, Delete in order
        for i,frequencies in enumerate(site_frequencies):
            length_difference = vector_length-len(frequencies)
            frequencies.extend([0]*length_difference)

            frequency_list.extend(frequencies)

        return frequency_list

    def _get_chromosome_number(self, chromosome_name):
        number = int(chromosome_name.split("chr")[-1])

        return number

    @staticmethod
    def split_frequencies_by_type(frequencies):
        delete_frequencies = list()
        insert_frequencies = list()
        mismatch_frequencies = list()

        split_frequencies = [mismatch_frequencies, insert_frequencies, delete_frequencies]

        for allele in frequencies:
            # print(allele)
            allele_tuple, count, frequency = allele
            allele_sequence, type = allele_tuple

            split_frequencies[type-1].append(count)

            # print(allele_sequence, type, frequency)

        split_frequencies = [sorted(frequency_list, reverse=True) for frequency_list in split_frequencies]

        return split_frequencies

    def _generate_data_vector(self, chromosome_name, start, frequencies, coverage, map_quality_non_ref, base_quality_non_ref, map_quality_ref, base_quality_ref):
        # split the list of allele frequency tuples by their type
        frequencies = self.split_frequencies_by_type(frequencies)

        data_list = list()
        header = list()

        # convert list of frequencies into vector with length 3*PLOIDY
        site_frequency_list = self._generate_fixed_size_freq_list(frequencies)

        chromosome_number = self._get_chromosome_number(chromosome_name)

        # # cap and normalize coverage (max = 1000)
        # coverage = min(coverage, 1000)
        # coverage = float(coverage)/1000

        data_list.append(chromosome_number)     # 0
        header.append("chromosome_number")

        data_list.append(int(start))            # 1
        header.append("position")

        data_list.extend(site_frequency_list)   # 4-15
        header.extend(["frequency_"+str(i+1) for i in range(len(site_frequency_list))])

        data_list.append(coverage)              # 16
        header.append("coverage")

        data_list.extend(map_quality_ref)       # 17-22
        header.extend(["map_quality_ref_"+str(i+1) for i in range(len(map_quality_ref))])

        data_list.extend(base_quality_ref)      # 23-28
        header.extend(["base_quality_ref_"+str(i+1) for i in range(len(base_quality_ref))])

        data_list.extend(map_quality_non_ref)   # 29-34
        header.extend(["map_quality_non_ref_"+str(i+1) for i in range(len(map_quality_non_ref))])

        data_list.extend(base_quality_non_ref)  # 35-42
        header.extend(["base_quality_non_ref_"+str(i+1) for i in range(len(base_quality_non_ref))])

        # convert data list to numpy Float vector
        data_row = pandas.DataFrame([data_list], columns=header)

        return data_row

    def normalize_data_table(self, data_table):
        """
        For the appropriate data types, normalize, cap, or downscale values based on column header names
        :param data_table:
        :param coverage:
        :return:
        """
        headers = data_table.columns.values

        # print(data_table)

        for header in headers:
            if header.startswith("frequency"):
                data_table[header] /= data_table["coverage"]

            if "quality" in header:
                data_table[header] /= QUALITY_SCALING_CONSTANT

            if header == "coverage":
                mask = data_table[header] > 1000
                data_table.loc[mask, header] = 1000
                data_table[header] /= 1000

        return data_table

    def get_vectorized_candidates(self, chromosome_name, candidate_sites):
        """
        Label candidates given variants from a VCF
        :param candidate_sites: Candidates with the format list of lists, where each sublist is data pertaining to a
        type, MISMATCH, INSERT, or DELETE:
            [['chr1', 10400, 10400, 'T', 'A', 'G', 'SUB', ['A', 'G', 'C'], [11, 11, 7], 500],
            ['chr1', 10400, 10400, 'T', 'TA', '.', 'IN', ['TA'], [4], 500],
            ['chr1', 10400, 10401, 'TA', 'T', '.', 'DEL', ['TA'], [6], 500]]
        :return: List of labeled candidate sites
        """

        # list of all labeled training vectors
        all_labeled_vectors = list()

        for candidate in candidate_sites:
            chr_name = candidate[CHR_NAME]
            allele_start = candidate[START]
            allele_stop = candidate[STOP]
            ref_sequence = candidate[REF_INDEX]
            alt1 = candidate[ALT1]
            alt2 = candidate[ALT2]
            alt1_type = candidate[ALT1_TYPE]
            alt2_type = candidate[ALT2_TYPE]
            alleles = list()
            alleles.append(alt1)
            alleles.append(alt2)
            alt_types = list()
            alt_types.append(alt1_type)
            alt_types.append(alt2_type)

            frequencies = candidate[FREQUENCIES]
            coverage = candidate[COVERAGE]

            map_quality_non_ref = candidate[MAP_QUALITY_NON_REF]
            base_quality_non_ref = candidate[BASE_QUALITY_NON_REF]
            map_quality_ref = candidate[MAP_QUALITY_REF]
            base_quality_ref = candidate[BASE_QUALITY_REF]

            alleles = [alt1, alt2]

            vector = self._generate_data_vector(chromosome_name=chromosome_name,
                                                start=allele_start,
                                                frequencies=frequencies,
                                                coverage=coverage,
                                                map_quality_non_ref=map_quality_non_ref,
                                                base_quality_non_ref=base_quality_non_ref,
                                                map_quality_ref=map_quality_ref,
                                                base_quality_ref=base_quality_ref)

            all_labeled_vectors.append(vector)

        # if there is no data for this region, append an empty vector
        # if len(all_labeled_vectors) == 0:
        #     data_table = pandas.DataFrame([])

        if not len(all_labeled_vectors) == 0:
            # concatenate
            data_table = pandas.concat(all_labeled_vectors)

            # normalize
            data_table = self.normalize_data_table(data_table=data_table)
        else:
            data_table = None

        # data_table.to_csv("test.tsv", sep='\t')

        return data_table
