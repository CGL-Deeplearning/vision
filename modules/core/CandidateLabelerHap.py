from operator import itemgetter
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
MAX_GROUP_SIZE = 8
MAX_SEPARATION = 30

GENOTYPE_DICT = {"Hom": 0, "Het": 1, "Hom_alt": 2}
VCF_OFFSET = 1
PLOIDY = 2

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

# Positional vcf indexes
SNP, IN, DEL = 0, 1, 2
SNP_CANDIDATE, IN_CANDIDATE, DEL_CANDIDATE = 1, 2, 3
SNP_DEL = 3

# VCF record indexes
REF, ALT, GT = 0, 1, 2

# Genotype codes
HOM, HET, HOM_ALT = 0, 1, 2


class CandidateLabeler:
    def __init__(self, fasta_handler, vcf_handler):
        """
        Initialize candidateLabeler object
        :param fasta_handler: module that fetches reference sequence substrings from a FASTA file
        """
        self.fasta_handler = fasta_handler      # unfortunately need to query reference sequence on a per-site basis

        self.vcf_offset = -1                    # pysam vcf coords are 1-based ... >:[ ... this is what Kishwar wanted
        self.delete_char = '*'
        self.vcf_handler = vcf_handler

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
            ref = record.ref
            rec_alt = record.alt
            if record.genotype_class == "SNP":
                ref = ref[0]
                rec_alt = rec_alt[0]

            if record.type == '':
                record.type = 'Hom'
            # if the alt allele of the record is same as candidate allele
            vcf_recs.append((ref, rec_alt, record.type, record.filter, record.mq, record.gq, record.gt))
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
        :return: Dictionary of genotypes
        """
        gts = self.get_label_of_allele(positional_vcf, (start, stop, ref_seq, alleles), allele_types)

        return gts

    @staticmethod
    def _is_supported(genotypes):
        """
        Check if genotype has anything other than Hom
        :param genotypes: Genotype tuple
        :return: Boolean [True if it has Het of Hom_alt]
        """
        supported = False
        gt_set = set(genotypes)

        if len(gt_set) == 0:
            supported = False
        elif len(gt_set) == 1:
            if HOM in gt_set:
                supported = False
            else:
                supported = True
        elif len(gt_set) > 1:
            supported = True

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
        :param alleles_snp: SNP alleles
        :param alleles_in: Insert alleles
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

    @staticmethod
    def get_combined_gt(gt1, gt2):
        """
        Given two genotypes get the combined genotype. This is used to create labels for the third image.
        If two alleles have two different genotypes then the third genotype is inferred using this method.

        - If genotype1 is HOM then genotype of third image is genotype2
        - If genotype2 is HOM then genotype of third image is genotype1
        - If both gt are  HOM then genotype of third image is HOM
        - If genotype1, genotype2 both are HET then genotype of third image is HOM_ALT
        - If none of these cases match then we have an invalid genotype
        :param gt1: Genotype of first allele
        :param gt2: Genotype of second allele
        :return: genotype of image where both alleles are used together
        """
        if gt1 == 0:
            return gt2
        if gt2 == 0:
            return gt1
        if gt1 == 0 and gt2 == 0:
            return 0
        if gt1 == 1 and gt2 == 1:
            return 2
        return None

    def group_variants(self, candidate_records, chromosome_name, pos_start, pos_end):
        truth_records = self.vcf_handler.get_simple_vcf_records(chromosome_name, pos_start, pos_end)
        combined_records = sorted(truth_records+candidate_records, key=itemgetter(1))

        groups = []

        current_group = []
        current_truth_count = 0
        current_candidate_count = 0
        last_end = -1
        for record in combined_records:
            current_record_is_candidate = record[6]
            record_pos = record[1]
            max_allele_length = max(len(record[2]), len(record[3]), len(record[4]))

            can_add = False
            if not current_group:
                can_add = True
            if current_record_is_candidate and current_candidate_count < MAX_GROUP_SIZE:
                can_add = True
            elif not current_record_is_candidate and current_truth_count < MAX_GROUP_SIZE:
                can_add = True
            elif last_end - record_pos + 1 <= MAX_SEPARATION:
                can_add = True

            if can_add:
                current_group.append(record)
                if current_record_is_candidate:
                    current_candidate_count += 1
                else:
                    current_truth_count += 1
                last_end = max(last_end, record_pos + max_allele_length)
            else:
                groups.append(current_group)

                current_group = [record]
                current_truth_count = 0
                current_candidate_count = 0

                last_end = record_pos + max_allele_length
                if current_record_is_candidate:
                    current_candidate_count += 1
                else:
                    current_truth_count += 1
        if current_group:
            groups.append(current_group)

        separated_groups = []
        for group in groups:
            candidates = [record for record in group if record[6] is False]
            truths = [record for record in group if record[6] is True]
            separated_groups.append((candidates, truths))

        return separated_groups

    def get_labeled_candidates(self, chromosome_name, pos_start, pos_end, candidate_sites):
        """
        Label candidates given variants from a VCF
        :param positional_vcf: IN/DEL/SNPs separated into VCF records, expanded into a 1-to-1 ref_pos:variant allele
        :param candidate_sites: Candidates
        :return: List of labeled candidate sites
        """
        # list of all labeled candidates
        all_labeled_candidates = []
        candidate_records = []
        for candidate in candidate_sites:
            candidate_record = (candidate[0], candidate[1], candidate[3], candidate[4], candidate[5], (0, 0), False)
            candidate_records.append(candidate_record)

        # sort candidates based on positions
        candidate_records = sorted(candidate_records, key=itemgetter(1))
        groups = self.group_variants(candidate_records, chromosome_name, pos_start, pos_end)

        #for candidate_group, truth_group in groups:
        #    ref = self.make_labeler_ref(candidates_group, truth_group)
        # exit()
        # for each candidate
        for candidate_site in candidate_sites:

            chr_name = candidate_site[CHR_NAME]
            allele_start = candidate_site[START]
            allele_stop = candidate_site[STOP]
            ref_sequence = candidate_site[REF_INDEX]
            alt1 = candidate_site[ALT1]
            alt2 = candidate_site[ALT2]
            alt1_type = candidate_site[ALT1_TYPE]
            alt2_type = candidate_site[ALT2_TYPE]
            alleles = list()
            alleles.append(alt1)
            alleles.append(alt2)
            alt_types = list()
            alt_types.append(alt1_type)
            alt_types.append(alt2_type)

            # test the alleles across IN, DEL, and SNP variant dictionaries
            genotypes = self._get_all_genotype_labels(positional_vcf=positional_vcf,
                                                      start=allele_start,
                                                      stop=allele_stop,
                                                      ref_seq=ref_sequence,
                                                      alleles=alleles,
                                                      allele_types=alt_types)

            # get a list of attributes that can be saved
            labeled_candidate = candidate_site + genotypes
            all_labeled_candidates.append(labeled_candidate)

            if DEBUG_PRINT_ALL:
                print()
                print(allele_start, allele_stop, ref_sequence)
                print("alleles:        ", alleles)
                print("Genotypes:     ", genotypes)

        return all_labeled_candidates
