import numpy
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

class CandidateLabeler:
    def __init__(self, fasta_handler):
        """
        Initialize candidateLabeler object
        :param fasta_handler: module that fetches reference sequence substrings from a FASTA file
        """
        self.fasta_handler = fasta_handler      # unfortunately need to query reference sequence on a per-site basis

        self.vcf_offset = -1                    # pysam vcf coords are 1-based ... >:[ ... this is what Kishwar wanted
        self.delete_char = '*'

    def _handle_insert(self, rec):
        """
        Process a record that has an insert
        :param rec: VCF record
        :return: attributes of the record
        """
        ref_seq = rec.ref   # no change necessary
        alt_seq = rec.alt   # no change necessary

        pos = rec.pos + self.vcf_offset
        return pos, ref_seq, alt_seq, rec.type

    def _handle_delete(self, rec):
        """
        Process a record that has deletes.
        Deletes are usually grouped together, so we break each of the deletes to make a list.
        :param rec: VCF record containing a delete
        :return: A list of delete attributes
        """
        delete_list = []
        for i in range(0, len(rec.ref)):
            if i < len(rec.alt):
                continue
            ref_seq = rec.ref[i]
            alt_seq = '*'
            pos = rec.pos + i + self.vcf_offset
            genotype = rec.type
            delete_list.append((pos, ref_seq, alt_seq, genotype))
        return delete_list

    @staticmethod
    def _resolve_suffix_for_insert(ref, alt):
        len_ref = len(ref) - 1
        if len_ref == 0:
            return ref, alt
        suffix_removed_alt = alt[:-len_ref]
        return ref[0], suffix_removed_alt

    @staticmethod
    def _resolve_suffix_for_delete(ref, alt):
        len_alt = len(alt) - 1
        if len_alt == 0:
            return ref, alt
        suffix_removed_ref = ref[:-len_alt]
        return suffix_removed_ref, alt[0]

    @staticmethod
    def get_label_of_allele(positional_vcf, candidate_allele, allele_types):
        """
        Given positional VCFs (IN, DEL, SNP), variant type and a candidate allele, return the try genotype.
        :param positional_vcf: Three dictionaries for each position
        :param candidate_allele: Candidate allele
        :param allele_types: Alt allele type: IN,DEL,SNP
        :return: genotype
        """
        # candidate attributes
        pos_start, pos_stop, ref, alts = candidate_allele
        # get all records of that position
        records = []
        if pos_start + VCF_OFFSET in positional_vcf.keys():
            records = positional_vcf[pos_start + VCF_OFFSET]
            records = [record for type in records for record in type]

        refined_alts = []
        for i, alt in enumerate(alts):
            ref_ret = ref
            alt_seq, alt_type = alt, allele_types[i]
            if alt_type == DEL_CANDIDATE:
                ref_ret, alt_seq = alt_seq, ref_ret
            refined_alts.append([ref_ret, alt_seq, alt_type])
        vcf_recs = []
        for record in records:
            # get the alt allele of the record
            rec_alt = record.alt

            if record.type == '':
                record.type = 'Hom'
            # if the alt allele of the record is same as candidate allele
            vcf_recs.append((record.ref, rec_alt, record.type, record.filter, record.mq, record.gq, record.gt))
        gts = list()
        for alt in refined_alts:
            ref_seq, alt_seq, alt_type = alt
            gt = [0, '.']
            for vcf_alt in vcf_recs:
                vcf_ref, vcf_alt, vcf_genotype, vcf_filter, vcf_mq, vcf_gq, vcf_gt = vcf_alt

                if ref_seq == vcf_ref and alt_seq == vcf_alt:
                    gt = [GENOTYPE_DICT[vcf_genotype], vcf_filter, vcf_mq, vcf_gq, vcf_gt]
            gts.append(gt)

        return gts

    def _get_all_genotype_labels(self, positional_vcf, start, stop, ref_seq, alleles, allele_types):
        """
        Create a list of dictionaries of 3 types of alleles that can be in a position.

        In each position there can be Insert allele, SNP or Del alleles.
        For total 6 alleles, this method returns 6 genotypes
        :param positional_vcf: VCF records of each position
        :param start: Allele start position
        :param stop: Allele stop position
        :param ref_seq: Reference sequence
        :return: list of genotypes
        """
        gts = self.get_label_of_allele(positional_vcf, (start, stop, ref_seq, alleles), allele_types)

        genotype_labels = [gt[0] for gt in gts]

        return genotype_labels

    @staticmethod
    def _is_supported(genotypes):
        """
        Check if genotype has anything other than Hom
        :param genotypes: Genotype tuple
        :return: Boolean [True if it has Het of Hom_alt]
        """
        supported = False

        if sum(genotypes) > 0:
            supported = True

        # print("code:", genotypes, "support:", supported)
        return supported

    def _is_position_supported(self, genotypes):
        """
        Check if a position has any genotype other than Hom
        :param genotypes: Genotypes list of that position
        :return: Boolean [True if it has Het or Hom_alt]
        """
        in_supported = self._is_supported(genotypes[IN])
        del_supported = self._is_supported(genotypes[DEL])
        snp_supported = self._is_supported(genotypes[SNP])

        return in_supported or del_supported or snp_supported

    def _generate_list(self, chromosome_name, start, stop, alleles_snp, alleles_in, ref_seq, genotypes):
        """
        Generate a list of attributes that can be saved of a labeled candidate
        :param chromosome_name: Name of chromosome
        :param start: Allele start position
        :param stop: Allele end position
        :param alleles: All alleles
        :param alleles_insert: Insert alleles
        :param ref_seq: reference Sequence
        :param genotypes: Genotypes
        :return: A list containing (chr start stop ref_seq alt1 alt2 gt1 gt2)
        """
        all_candidates = []
        for i, allele_tuple in enumerate(alleles_snp):
            allele, freq = allele_tuple
            gt = genotypes[SNP][i][0]
            gt_q = genotypes[SNP][i][1]
            gt_f = genotypes[SNP][i][2]
            all_candidates.append([chromosome_name, start, stop, ref_seq, allele, gt, gt_q, gt_f])

        for i, allele_tuple in enumerate(alleles_in):
            allele, freq = allele_tuple
            gt = genotypes[IN][i][0]
            gt_q = genotypes[IN][i][1]
            gt_f = genotypes[IN][i][2]
            all_candidates.append([chromosome_name, start, stop, ref_seq, allele, gt, gt_q, gt_f])

        return all_candidates

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
            allele_tuple, count, frequency = allele
            allele_sequence, type = allele_tuple

            split_frequencies[type-1].append(count)

        split_frequencies = [sorted(frequency_list, reverse=True) for frequency_list in split_frequencies]

        return split_frequencies

    def _generate_data_vector(self, chromosome_name, start, genotypes, frequencies, coverage, map_quality_non_ref, base_quality_non_ref, map_quality_ref, base_quality_ref, support):
        # split the list of allele frequency tuples by their type
        frequencies = self.split_frequencies_by_type(frequencies)

        data_list = list()
        header = list()

        # convert list of frequencies into vector with length 3*PLOIDY
        site_frequency_list = self._generate_fixed_size_freq_list(frequencies)

        chromosome_number = self._get_chromosome_number(chromosome_name)
        label = int(support)

        # cap and scale down coverage (max = 1000)
        # coverage = min(coverage, 1000)
        # coverage = float(coverage)/1000

        # scale down qualities
        # map_quality_ref = [mq/1000 for mq in map_quality_ref]
        # base_quality_ref = [bq/1000 for bq in base_quality_ref]
        # map_quality_non_ref = [mq/1000 for mq in map_quality_non_ref]
        # base_quality_non_ref = [bq/1000 for bq in base_quality_non_ref]

        data_list.append(chromosome_number)     # 0
        header.append("chromosome_number")

        data_list.append(int(start))            # 1
        header.append("position")

        data_list.extend(genotypes)             # 2-3
        header.extend(["genotype_"+str(i+1) for i in range(len(genotypes))])

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

        data_list.append(label)                 # 43
        header.append("label")

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

    def get_labeled_candidates(self, chromosome_name, positional_vcf, candidate_sites):
        """
        Label candidates given variants from a VCF
        :param positional_vcf: IN/DEL/SNPs separated into VCF records, expanded into a 1-to-1 ref_pos:variant allele
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

            # test the alleles across IN, DEL, and SNP variant dictionaries
            genotype = self._get_all_genotype_labels(positional_vcf=positional_vcf,
                                                     start=allele_start,
                                                     stop=allele_stop,
                                                     ref_seq=ref_sequence,
                                                     alleles=alleles,
                                                     allele_types=alt_types)

            support = self._is_supported(genotype)

            vector = self._generate_data_vector(chromosome_name=chromosome_name,
                                                start=allele_start,
                                                genotypes=genotype,
                                                frequencies=frequencies,
                                                coverage=coverage,
                                                support=support,
                                                map_quality_non_ref=map_quality_non_ref,
                                                base_quality_non_ref=base_quality_non_ref,
                                                map_quality_ref=map_quality_ref,
                                                base_quality_ref=base_quality_ref)

            all_labeled_vectors.append(vector)

        # # if there is no data for this region, append an empty vector
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
