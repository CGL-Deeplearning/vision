from pysam import VariantFile, VariantHeader
from collections import defaultdict
from modules.handlers.BamHandler import BamHandler
import math
import time
import numpy as np


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
    def get_genotype_tuple(genotype):
        split_values = genotype.split('/')
        split_values = [int(x) for x in split_values]
        return tuple(split_values)

    @staticmethod
    def get_genotype_for_single_allele(records):
        for record in records:
            probs = [record[7], record[8], record[9]]
            genotype_list = ['0/0', '0/1', '1/1']
            gq, index = max([(v, i) for i, v in enumerate(probs)])
            qual = sum(probs) - probs[0]

            phred_qual = min(60, -10 * np.log10(1 - qual) if 1 - qual >= 0.0000001 else 60)
            phred_qual = math.ceil(phred_qual * 100.0) / 100.0
            phred_gq = min(60, -10 * np.log10(1 - gq) if 1 - gq >= 0.0000001 else 60)
            phred_gq = math.ceil(phred_gq * 100.0) / 100.0

            return record[0], record[1], record[2], record[3], [record[4]], genotype_list[index], phred_qual, phred_gq

    @staticmethod
    def get_genotype_for_multiple_allele(records):
        ref = '.'
        st_pos = 0
        end_pos = 0
        chrm = ''
        rec_alt1 = '.'
        rec_alt2 = '.'
        alt_probs = defaultdict(list)
        for record in records:
            chrm = record[0]
            ref = record[3]
            st_pos = record[1]
            end_pos = record[2]
            alt1 = record[4]
            alt2 = record[5]
            if alt1 != '.' and alt2 != '.':
                rec_alt1 = alt1
                rec_alt2 = alt2
                alt_probs['both'] = (record[7:])
            else:
                alt_probs[alt1] = (record[7:])

        p00 = min(alt_probs[rec_alt1][0], alt_probs[rec_alt2][0], alt_probs['both'][0])
        p01 = min(alt_probs[rec_alt1][1], alt_probs['both'][1])
        p11 = min(alt_probs[rec_alt1][2], alt_probs['both'][2])
        p02 = min(alt_probs[rec_alt2][1], alt_probs['both'][1])
        p22 = min(alt_probs[rec_alt2][2], alt_probs['both'][2])
        p12 = min(max(alt_probs[rec_alt1][1], alt_probs[rec_alt1][2]),
                  max(alt_probs[rec_alt2][1], alt_probs[rec_alt2][2]),
                  max(alt_probs['both'][1], alt_probs['both'][2]))
        prob_list = [p00, p01, p11, p02, p22, p12]

        sum_probs = sum(prob_list)

        normalized_list = [(float(i) / sum_probs) if sum_probs else 0 for i in prob_list]
        prob_list = normalized_list

        genotype_list = ['0/0', '0/1', '1/1', '0/2', '2/2', '1/2']
        gq, index = 0, 0
        for i, prob in enumerate(prob_list):
            if gq <= prob and prob > 0:
                index = i
                gq = prob
        qual = sum(prob_list) - prob_list[0]

        phred_qual = min(60, -10 * np.log10(1 - qual) if 1 - qual >= 0.0000001 else 60)
        phred_qual = math.ceil(phred_qual * 100.0) / 100.0
        phred_gq = min(60, -10 * np.log10(1 - gq) if 1 - gq >= 0.0000001 else 60)
        phred_gq = math.ceil(phred_gq * 100.0) / 100.0

        return chrm, st_pos, end_pos, ref, [rec_alt1, rec_alt2], genotype_list[index], phred_qual, phred_gq

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

        record = chrm, st_pos, end_pos, ref, refined_alt, refined_gt, phred_qual, phred_gq

        return record

    @staticmethod
    def get_filter(record, last_end):
        chrm, st_pos, end_pos, ref, alt_field, genotype, phred_qual, phred_gq = record
        if st_pos <= last_end:
            return 'conflictPos'
        if genotype == '0/0':
            return 'refCall'
        if phred_qual < 0:
            return 'lowQUAL'
        if phred_gq < 0:
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
