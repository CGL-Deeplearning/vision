from modules.handlers.VcfHandler import VCFFileProcessor
from modules.handlers.FastaHandler import FastaHandler
from collections import defaultdict
import math
import random

PLOIDY = 2
VCF_OFFSET = 1
REF = 0
SNP = 1
INS = 2
DEL = 3

# vcf record indices
VCF_SNP, VCF_INS, VCF_DEL = 0, 1, 2

# allele object indices
SEQ, COUNT, INDEX = 0, 1, 2

READ_IDS = ["HAPLOTYPE1", "HAPLOTYPE2"]

# POS, QUALITY, REF_SEQ, ALT_SEQ, ZYGOSITY, FILTER, ALLELE_TUPLE = 0,1,2,3,4,5,6


class AlignmentGraphLabeler:
    def __init__(self, chromosome_name, start_position, end_position, reference_sequence, positional_variants, graph):
        self.start_position = start_position
        self.end_position = end_position
        self.chromosome_name = chromosome_name
        self.reference_sequence = reference_sequence
        self.positional_variants = positional_variants
        self.positional_genotypes = defaultdict(list)
        self.positional_alleles = defaultdict(lambda: [list(),list(),list(),list()])
        self.graph = graph

    def test_position_for_VCF_conflicts(self, position):
        gt_set = set()
        for genotype in self.positional_genotypes[position]:
            if len(gt_set) == 0:
                gt_set.add(genotype[0])
                gt_set.add(genotype[1])

            if genotype[0] not in gt_set or genotype[1] not in gt_set:
                print("WARNING: Non compatible genotypes found at position:", position, self.positional_genotypes[position])

    def test_position_for_ref_allele(self, position):
        # print(self.positional_genotypes[position])

        contains_ref_allele = False
        for gt in self.positional_genotypes[position]:
            print(gt)
            gt1, gt2 = gt
            if gt1 == 0 or gt2 == 0:
                contains_ref_allele = True

        # print(contains_ref_allele)
        return contains_ref_allele

    def get_n_alleles_from_zygosity(self, zygosity):
        if zygosity == "Het":
            n = 1
        else:
            n = 2

        return n

    def preprocess_mismatch(self, position, ref_sequence, alt_sequence, haplotype_index, genotype, zygosity):
        ref_allele = ref_sequence
        alt_allele = alt_sequence
        adjusted_position = position - VCF_OFFSET

        n = self.get_n_alleles_from_zygosity(zygosity)
        allele = (alt_allele, n, haplotype_index)

        self.positional_genotypes[adjusted_position].append(genotype)
        self.positional_alleles[adjusted_position][SNP].append(allele)

    def preprocess_insert(self, position, ref_sequence, alt_sequence, haplotype_index, genotype, zygosity):
        ref_allele = ref_sequence
        alt_allele = alt_sequence[1:]
        adjusted_position = position - VCF_OFFSET

        n = self.get_n_alleles_from_zygosity(zygosity)
        allele = (alt_allele, n, haplotype_index)

        self.positional_genotypes[adjusted_position].append(genotype)
        self.positional_alleles[adjusted_position][INS].append(allele)

    def preprocess_delete(self, position, ref_sequence, alt_sequence, haplotype_index, genotype, zygosity):
        ref_length = len(ref_sequence)
        ref_allele = ref_sequence[0]
        alt_allele = "*"

        n = self.get_n_alleles_from_zygosity(zygosity)
        allele = (alt_allele, n, haplotype_index)

        for i in range(1, ref_length):
            adjusted_position = position + i - VCF_OFFSET

            self.positional_genotypes[adjusted_position].append(genotype)
            self.positional_alleles[adjusted_position][DEL].append(allele)

    def print_positional_alleles(self):
        for position in self.positional_alleles:
            print(position,
                  "REF:", list(self.positional_alleles[position][REF]),
                  "SNP:", list(self.positional_alleles[position][SNP]),
                  "INS:", list(self.positional_alleles[position][INS]),
                  "DEL:", list(self.positional_alleles[position][DEL]))

    def preprocess_positional_variants(self):
        for position in self.positional_variants:
            alt_haplotype_index = 1

            variant_record = self.positional_variants[position]
            variant_types = list()
            n_hets = 0

            for variant_code in [VCF_SNP, VCF_INS, VCF_DEL]:
                for variant in variant_record[variant_code]:
                    # get variant attributes
                    zygosity = variant.type
                    alt_sequence = variant.alt
                    ref_sequence = variant.ref
                    genotype = variant.genotype

                    variant_types.append(variant_code)

                    alt_haplotype_index -= n_hets
                    if zygosity == "Het":
                        n_hets += 1

                    if variant_code == VCF_SNP:
                        self.preprocess_mismatch(position=position,
                                                 ref_sequence=ref_sequence,
                                                 alt_sequence=alt_sequence,
                                                 haplotype_index=alt_haplotype_index,
                                                 genotype=genotype,
                                                 zygosity=zygosity)

                    if variant_code == VCF_INS:
                        self.preprocess_insert(position=position,
                                               ref_sequence=ref_sequence,
                                               alt_sequence=alt_sequence,
                                               haplotype_index=alt_haplotype_index,
                                               genotype=genotype,
                                               zygosity=zygosity)

                    if variant_code == VCF_DEL:
                        self.preprocess_delete(position=position,
                                               ref_sequence=ref_sequence,
                                               alt_sequence=alt_sequence,
                                               haplotype_index=alt_haplotype_index,
                                               genotype=genotype,
                                               zygosity=zygosity)

            if len(self.positional_genotypes[position]) == 0:
                self.positional_genotypes[position].append((0,0))

            self.test_position_for_VCF_conflicts(position)

    def get_other_haplotype(self, haplotype_index):
        other_haplotype_index = abs(1-haplotype_index)

        return other_haplotype_index

    def flag_node_as_variant(self, position, cigar_code, sequence):
        if sequence in self.graph.graph[position][cigar_code]:
            self.graph.graph[position][cigar_code][sequence].true_variant = True
        else:
            print("WARNING: VCF node not found in BAM:", position, cigar_code, sequence)

    def parse_region(self):
        self.preprocess_positional_variants()

        for i,position in enumerate(range(self.start_position, self.end_position+1)):
            reference_sequence = self.reference_sequence[i]

            if position in self.positional_alleles:
                # get cigar data, sequences (up to 2 ploidy for human), and update the graph for each true allele

                haplotype_set = {0,1}
                insert_found = len(self.positional_alleles[position][INS]) > 0

                if insert_found:
                    inserts = self.positional_alleles[position][INS]

                    for insert_allele in inserts:
                        allele_sequence, n_alleles, haplotype_index = insert_allele

                        for n in range(n_alleles):
                            if n > 0:
                                haplotype_index = self.get_other_haplotype(haplotype_index)

                            # print(n, haplotype_index)
                        self.flag_node_as_variant(position=position, cigar_code=REF, sequence=reference_sequence)

                for cigar_code in [SNP, INS, DEL]:
                    for allele in self.positional_alleles[position][cigar_code]:
                        allele_sequence, n_alleles, haplotype_index = allele

                        for n in range(n_alleles):
                            if n > 0:
                                haplotype_index = self.get_other_haplotype(haplotype_index)

                            self.flag_node_as_variant(position=position, cigar_code=cigar_code, sequence=allele_sequence)

                            haplotype_set.remove(haplotype_index)

                # if all haplotypes have not been used for this position, there must be a reference allele
                for haplotype_index in haplotype_set:
                    self.flag_node_as_variant(position=position, cigar_code=REF, sequence=reference_sequence)

            else:
                # update using reference allele for both imaginary haplotypes
                self.flag_node_as_variant(position=position, cigar_code=REF, sequence=reference_sequence)
