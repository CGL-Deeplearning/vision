from pysam import VariantFile, VariantHeader
from collections import defaultdict
from modules.handlers.BamHandler import BamHandler
import math
import time
import numpy as np

DEL_TYPE = '3'
IN_TYPE = '2'
SNP_TYPE = '1'

QUAL_FILTER = 5.0
GQ_FILTER = 2.0
MULTI_ALLELE_QUAL_FILTER = 7.0
SINGLE_ALLELE_SNP_QUAL_FILTER = 5.0
SINGLE_ALLELE_INDEL_QUAL_FILTER = 6.0


class VCFWriter:
    def __init__(self, bam_file_path, sample_name, output_dir):
        self.bam_handler = BamHandler(bam_file_path)
        bam_file_name = bam_file_path.rstrip().split('/')[-1].split('.')[0]
        vcf_header = self.get_vcf_header(sample_name)
        time_str = time.strftime("%m%d%Y_%H%M%S")
        self.vcf_file = VariantFile(output_dir + bam_file_name + '_' + time_str + '.vcf', 'w', header=vcf_header)

    def write_vcf_record(self, chrm, st_pos, end_pos, ref, alts, genotype, qual, gq, rec_filter):
        alleles = tuple([ref]) + tuple(alts)
        genotype = self.get_genotype_tuple(genotype)
        end_pos = int(end_pos) + 1
        st_pos = int(st_pos)
        vcf_record = self.vcf_file.new_record(contig=chrm, start=st_pos, stop=end_pos, id='.', qual=qual,
                                              filter=rec_filter, alleles=alleles, GT=genotype, GQ=gq)
        self.vcf_file.write(vcf_record)

    @staticmethod
    def solve_multiple_alts(alts, ref):
        type1, type2 = alts[0][1], alts[1][1]
        alt1, alt2 = alts[0][0], alts[1][0]
        if type1 == DEL_TYPE and type2 == DEL_TYPE:
            if len(alt2) > len(alt1):
                return alt2, ref, alt2[0] + alt2[len(alt1):]
            else:
                return alt1, ref, alt1[0] + alt1[len(alt2):]
        elif type1 == IN_TYPE and type2 == IN_TYPE:
            return ref, alt1, alt2
        elif type1 == DEL_TYPE or type2 == DEL_TYPE:
            if type1 == DEL_TYPE and type2 == IN_TYPE:
                return alt1, ref, alt2 + alt1[1:]
            elif type1 == IN_TYPE and type2 == DEL_TYPE:
                return alt2, alt1 + alt2[1:], ref
            elif type1 == DEL_TYPE and type2 == SNP_TYPE:
                return alt1, ref, alt2 + alt1[1:]
            elif type1 == SNP_TYPE and type2 == DEL_TYPE:
                return alt2, alt1 + alt2[1:], ref
            elif type1 == DEL_TYPE:
                return alt1, ref, alt2
            elif type2 == DEL_TYPE:
                return alt2, alt1, ref
        else:
            return ref, alt1, alt2

    @staticmethod
    def solve_single_alt(alts, ref):
        # print(alts)
        alt1, alt_type = alts
        if alt_type == DEL_TYPE:
            return alt1, ref, '.'
        return ref, alt1, '.'

    @staticmethod
    def get_genotype_tuple(genotype):
        split_values = genotype.split('/')
        split_values = [int(x) for x in split_values]
        return tuple(split_values)

    @staticmethod
    def get_genotype_for_single_allele(record):
        chromosome_name, pos_start, pos_end, ref, alt1, alt2, alt1_type, alt2_type = record[0:8]
        alt_type = int(alt1_type)
        probabilities = [float(record[8]), float(record[9]), float(record[10])]

        genotype_list = ['0/0', '0/1', '1/1']
        qual = sum(probabilities) - probabilities[0]
        phred_qual = min(60, -10 * np.log10(1 - qual) if 1 - qual >= 0.0000001 else 60)

        alt_with_types = list()
        alt_with_types.append((alt1, alt1_type))
        ref, alt1, alt2 = VCFWriter.solve_single_alt(alt_with_types[0], ref)

        # it's a SNP
        if alt_type == 1:
            if phred_qual < SINGLE_ALLELE_SNP_QUAL_FILTER:
                index = 0
                gq = probabilities[0]
            else:
                gq, index = max([(v, i) for i, v in enumerate(probabilities)])
        else:
            # in an indel
            if phred_qual < SINGLE_ALLELE_INDEL_QUAL_FILTER:
                index = 0
                gq = probabilities[0]
            else:
                gq, index = max([(v, i) for i, v in enumerate(probabilities)])
        phred_gq = min(60, -10 * np.log10(1 - gq) if 1 - gq >= 0.0000001 else 60)

        return chromosome_name, pos_start, pos_end, ref, [alt1, alt2], genotype_list[index], phred_qual, phred_gq

    @staticmethod
    def get_genotype_for_multiple_allele(records):
        ref = '.'
        st_pos = 0
        end_pos = 0
        chrm = ''
        rec_alt1 = '.'
        rec_alt2 = '.'
        alt_probs = defaultdict(tuple)
        alt_with_types = []

        for record in records:
            chrm, st_pos, end_pos, ref, alt1, alt2, alt1_type, alt2_type = record[0:8]
            probs = [float(record[8]), float(record[9]), float(record[10])]
            if alt1 != '.' and alt2 != '.':
                rec_alt1 = (alt1, alt1_type)
                rec_alt2 = (alt2, alt2_type)
                alt_probs['both'] = probs
            else:
                alt_probs[(alt1, alt1_type)] = probs
                alt_with_types.append((alt1, alt1_type))

        p00 = min(alt_probs[rec_alt1][0], alt_probs[rec_alt2][0], alt_probs['both'][0])
        p01 = min(alt_probs[rec_alt1][1], alt_probs['both'][1])
        p11 = min(alt_probs[rec_alt1][2], alt_probs['both'][2])
        p02 = min(alt_probs[rec_alt2][1], alt_probs['both'][1])
        p22 = min(alt_probs[rec_alt2][2], alt_probs['both'][2])
        p12 = alt_probs['both'][2]

        # print(alt_probs)
        alt1_probs = [p00, p01, p11]
        alt1_norm_probs = [(float(prob) / sum(alt1_probs)) if sum(alt1_probs) else 0 for prob in alt1_probs]
        alt1_qual = sum(alt1_norm_probs) - alt1_norm_probs[0]

        alt2_probs = [p00, p02, p22]
        alt2_norm_probs = [(float(prob) / sum(alt2_probs)) if sum(alt2_probs) else 0 for prob in alt2_probs]
        alt2_qual = sum(alt2_norm_probs) - alt2_norm_probs[0]

        alt1_phred_qual = min(60, -10 * np.log10(1 - alt1_qual) if 1 - alt1_qual >= 0.0000001 else 60)
        alt2_phred_qual = min(60, -10 * np.log10(1 - alt2_qual) if 1 - alt2_qual >= 0.0000001 else 60)

        if alt1_phred_qual < MULTI_ALLELE_QUAL_FILTER and alt2_phred_qual < MULTI_ALLELE_QUAL_FILTER:
            if alt1_phred_qual > alt2_phred_qual:
                probs = [p00, p01, p11]
                sum_probs = sum(probs)
                probs = [(float(i) / sum_probs) if sum_probs else 0 for i in probs]

                genotype_list = ['0/0', '0/1', '1/1']
                gq, index = max([(v, i) for i, v in enumerate(probs)])
                qual = sum(probs) - probs[0]
                phred_qual = min(60, -10 * np.log10(1 - qual) if 1 - qual >= 0.0000001 else 60)
                phred_gq = min(60, -10 * np.log10(1 - gq) if 1 - gq >= 0.0000001 else 60)

                ref, alt1, alt2 = VCFWriter.solve_single_alt(rec_alt1, ref)

                return chrm, st_pos, end_pos, ref, [alt1, alt2], genotype_list[index], phred_qual, phred_gq
            else:
                probs = [p00, p02, p22]
                sum_probs = sum(probs)
                probs = [(float(i) / sum_probs) if sum_probs else 0 for i in probs]

                genotype_list = ['0/0', '0/2', '2/2']
                gq, index = max([(v, i) for i, v in enumerate(probs)])
                qual = sum(probs) - probs[0]
                phred_qual = min(60, -10 * np.log10(1 - qual) if 1 - qual >= 0.0000001 else 60)
                phred_gq = min(60, -10 * np.log10(1 - gq) if 1 - gq >= 0.0000001 else 60)

                ref, alt2, alt1 = VCFWriter.solve_single_alt(rec_alt2, ref)

                return chrm, st_pos, end_pos, ref, [alt1, alt2], genotype_list[index], phred_qual, phred_gq
        elif alt1_phred_qual < MULTI_ALLELE_QUAL_FILTER:
            probs = [p00, p02, p22]
            sum_probs = sum(probs)
            probs = [(float(i) / sum_probs) if sum_probs else 0 for i in probs]

            genotype_list = ['0/0', '0/2', '2/2']
            gq, index = max([(v, i) for i, v in enumerate(probs)])
            qual = sum(probs) - probs[0]
            phred_qual = min(60, -10 * np.log10(1 - qual) if 1 - qual >= 0.0000001 else 60)
            phred_gq = min(60, -10 * np.log10(1 - gq) if 1 - gq >= 0.0000001 else 60)

            ref, alt2, alt1 = VCFWriter.solve_single_alt(rec_alt2, ref)

            return chrm, st_pos, end_pos, ref, [alt1, alt2], genotype_list[index], phred_qual, phred_gq
        elif alt2_phred_qual < MULTI_ALLELE_QUAL_FILTER:
            probs = [p00, p01, p11]
            sum_probs = sum(probs)
            probs = [(float(i) / sum_probs) if sum_probs else 0 for i in probs]

            genotype_list = ['0/0', '0/1', '1/1']
            gq, index = max([(v, i) for i, v in enumerate(probs)])
            qual = sum(probs) - probs[0]
            phred_qual = min(60, -10 * np.log10(1 - qual) if 1 - qual >= 0.0000001 else 60)
            phred_gq = min(60, -10 * np.log10(1 - gq) if 1 - gq >= 0.0000001 else 60)

            ref, alt1, alt2 = VCFWriter.solve_single_alt(rec_alt1, ref)

            return chrm, st_pos, end_pos, ref, [alt1, alt2], genotype_list[index], phred_qual, phred_gq
        else:
            prob_list = [p00, p01, p11, p02, p22, p12]
            sum_probs = sum(prob_list)
            # print(sum_probs)
            normalized_list = [(float(i) / sum_probs) if sum_probs else 0 for i in prob_list]
            prob_list = normalized_list
            # print(prob_list)
            # print(sum(prob_list))
            genotype_list = ['0/0', '0/1', '1/1', '0/2', '2/2', '1/2']
            gq, index = max([(v, i) for i, v in enumerate(prob_list)])
            qual = sum(prob_list) - prob_list[0]
            phred_qual = min(60, -10 * np.log10(1 - qual) if 1 - qual >= 0.0000001 else 60)
            phred_gq = min(60, -10 * np.log10(1 - gq) if 1 - gq >= 0.0000001 else 60)

            ref, alt1, alt2 = VCFWriter.solve_multiple_alts([rec_alt1, rec_alt2], ref)
            return chrm, st_pos, end_pos, ref, [alt1, alt2], genotype_list[index], phred_qual, phred_gq

    @staticmethod
    def get_proper_alleles(record):
        chrm, st_pos, end_pos, ref, alt_field, genotype, phred_qual, phred_gq = record
        gts = genotype.split('/')
        refined_alt = []
        refined_gt = genotype
        if gts[0] == '1' or gts[1] == '1':
            refined_alt.append(alt_field[0])
        if gts[0] == '2' or gts[1] == '2':
            refined_alt.append(alt_field[1])
        if gts[0] == '0' and gts[1] == '0':
            refined_alt.append('.')
        if genotype == '0/2':
            refined_gt = '0/1'
        if genotype == '2/2':
            refined_gt = '1/1'

        end_pos = st_pos + len(ref) - 1
        record = chrm, st_pos, end_pos, ref, refined_alt, refined_gt, phred_qual, phred_gq

        return record

    @staticmethod
    def get_filter(record, last_end):
        chrm, st_pos, end_pos, ref, alt_field, genotype, phred_qual, phred_gq = record
        if st_pos <= last_end:
            return 'conflictPos'
        if genotype == '0/0':
            return 'refCall'
        if phred_qual <= QUAL_FILTER:
            return 'lowQUAL'
        if phred_gq <= GQ_FILTER:
            return 'lowGQ'
        return 'PASS'

    def get_vcf_header(self, sample_name):
        header = VariantHeader()
        items = [('ID', "PASS"),
                 ('Description', "All filters passed")]
        header.add_meta(key='FILTER', items=items)
        items = [('ID', "refCall"),
                 ('Description', "Call is homozygous")]
        header.add_meta(key='FILTER', items=items)
        items = [('ID', "lowGQ"),
                 ('Description', "Low genotype quality")]
        header.add_meta(key='FILTER', items=items)
        items = [('ID', "lowQUAL"),
                 ('Description', "Low variant call quality")]
        header.add_meta(key='FILTER', items=items)
        items = [('ID', "conflictPos"),
                 ('Description', "Overlapping record")]
        header.add_meta(key='FILTER', items=items)
        items = [('ID', "GT"),
                 ('Number', 1),
                 ('Type', 'String'),
                 ('Description', "Genotype")]
        header.add_meta(key='FORMAT', items=items)
        items = [('ID', "GQ"),
                 ('Number', 1),
                 ('Type', 'Float'),
                 ('Description', "Genotype Quality")]
        header.add_meta(key='FORMAT', items=items)
        bam_sqs = self.bam_handler.get_header_sq()
        for sq in bam_sqs:
            id = sq['SN']
            ln = sq['LN']
            items = [('ID', id),
                     ('length', ln)]
            header.add_meta(key='contig', items=items)

        header.add_sample(sample_name)

        return header
