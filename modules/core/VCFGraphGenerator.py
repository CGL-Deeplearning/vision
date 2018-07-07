from modules.handlers.VcfHandler import VCFFileProcessor
from modules.handlers.FastaHandler import FastaHandler
import random

VCF_OFFSET = 1
REF = 0
SNP = 1
INS = 2
DEL = 3

VCF_SNP, VCF_INS, VCF_DEL = 0, 1, 2

READ_IDS = ["HAPLOTYPE1", "HAPLOTYPE2"]

# POS, QUALITY, REF_SEQ, ALT_SEQ, ZYGOSITY, FILTER, ALLELE_TUPLE = 0,1,2,3,4,5,6


class VCFGraphGenerator:
    def __init__(self, chromosome_name, start_position, end_position, reference_sequence, positional_variants, graph):
        self.start_position = start_position
        self.end_position = end_position
        self.chromosome_name = chromosome_name
        self.reference_sequence = reference_sequence
        self.positional_variants = positional_variants
        self.graph = graph

        self.vcf_offset = 1

    def parse_region(self):
        for i,position in enumerate(range(self.start_position, self.end_position+1)):
            reference_sequence = self.reference_sequence[i]

            if position + VCF_OFFSET in self.positional_variants:
                # get cigar data, sequences (up to 2 ploidy for human), and update the graph for each true allele
                print("VARIANT at position", position, self.positional_variants[position+VCF_OFFSET])

                self.parse_positional_variants(position=position+VCF_OFFSET)

            else:
                # update with reference allele (once per imaginary haplotype)
                self.graph.update_position(read_id=READ_IDS[0],
                                           position=position,
                                           sequence=reference_sequence,
                                           cigar_code=REF)

                self.graph.update_position(read_id=READ_IDS[1],
                                           position=position,
                                           sequence=reference_sequence,
                                           cigar_code=REF)

    def parse_mismatch(self, position, alt_sequence, haplotype_index):
        self.graph.update_position(read_id=READ_IDS[haplotype_index],
                                   position=position,
                                   sequence=alt_sequence,
                                   cigar_code=SNP)

    def parse_insert(self, position, alt_sequence, haplotype_index):
        self.graph.update_position(read_id=READ_IDS[haplotype_index],
                                   position=position,
                                   sequence=alt_sequence,
                                   cigar_code=INS)

    def parse_delete(self, position, alt_sequence, ref_sequence, haplotype_index):
        # --------------------------
        # HOW TO DEAL WITH DELETES??
        # --------------------------
        length = len(ref_sequence) - len(alt_sequence)

        for offset in range(length):
            self.graph.update_position(read_id=READ_IDS[haplotype_index],
                                       position=position,
                                       sequence=alt_sequence,
                                       cigar_code=DEL)

    def update_reference_at_variant_position(self, position, genotypes, reference_sequence, haplotype_index):
        reference_allele_exists = False

        for gt_pair in genotypes:
            for gt in gt_pair:
                if int(gt) == REF:
                    reference_allele_exists = True

        if reference_allele_exists:
            self.graph.update_position(read_id=random.choice(READ_IDS[haplotype_index]),
                                       position=position,
                                       sequence=reference_sequence,
                                       cigar_code=REF)

    def parse_positional_variants(self, position):
        ref_haplotype_index = 0
        alt_haplotype_index = 1

        ref_sequence = None
        variant_record = self.positional_variants[position]
        genotypes = list()
        variant_types = list()
        n_hets = 0

        for variant_code in [VCF_SNP, VCF_INS, VCF_DEL]:
            for variant in variant_record[variant_code]:
                # get variant attributes
                zygosity = variant.type
                alt_sequence = variant.alt
                ref_sequence = variant.ref
                genotype = variant.genotype

                genotypes.append(genotype)
                variant_types.append(variant_code)

                print(position, "zyg:", zygosity, "alt_seq:", alt_sequence, "ref_seq:", ref_sequence)

                alt_haplotype_index -= n_hets
                if zygosity == "Het":
                    n_hets += 1

                if variant_code == VCF_SNP:
                    self.parse_mismatch(position=position-VCF_OFFSET,
                                        alt_sequence=alt_sequence,
                                        haplotype_index=alt_haplotype_index)

                if variant_code == VCF_INS:
                    self.parse_insert(position=position-VCF_OFFSET,
                                      alt_sequence=alt_sequence,
                                      haplotype_index=alt_haplotype_index)

                if variant_code == VCF_DEL:
                    self.parse_delete(position=position-VCF_OFFSET,
                                      alt_sequence=alt_sequence,
                                      haplotype_index=alt_haplotype_index)

        print(genotypes)

        self.update_reference_at_variant_position(position=position-VCF_OFFSET,
                                                  genotypes=genotypes,
                                                  reference_sequence=reference_sequence,
                                                  haplotype_index=ref_haplotype_index)


        # fetch alleles from positional variant dictionary
        # pos_start, pos_stop, ref, alts = candidate_allele
        # # get all records of that position
        # records = []
        # if pos_start + VCF_OFFSET in positional_vcf.keys():
        #     records = positional_vcf[pos_start + VCF_OFFSET]
        #     records = [record for type in records for record in type]
        #
        # refined_alts = []
        # for i, alt in enumerate(alts):
        #     ref_ret = ref
        #     alt_seq, alt_type = alt, allele_types[i]
        #     if alt_type == DEL_CANDIDATE:
        #         ref_ret, alt_seq = alt_seq, ref_ret
        #     refined_alts.append([ref_ret, alt_seq, alt_type])
        # vcf_recs = []
        # for record in records:
        #     # get the alt allele of the record
        #     rec_alt = record.alt
        #
        #     if record.type == '':
        #         record.type = 'Hom'
        #     # if the alt allele of the record is same as candidate allele
        #     vcf_recs.append((record.ref, rec_alt, record.type, record.filter, record.mq, record.gq, record.gt))
        # gts = list()
        # for alt in refined_alts:
        #     ref_seq, alt_seq, alt_type = alt
        #     gt = [0, '.']
        #     for vcf_alt in vcf_recs:
        #         vcf_ref, vcf_alt, vcf_genotype, vcf_filter, vcf_mq, vcf_gq, vcf_gt = vcf_alt
        #
        #         if ref_seq == vcf_ref and alt_seq == vcf_alt:
        #             gt = [GENOTYPE_DICT[vcf_genotype], vcf_filter, vcf_mq, vcf_gq, vcf_gt]
        #     gts.append(gt)
        #
        # return gts

        # parse as though parsing operations in candidate finder

