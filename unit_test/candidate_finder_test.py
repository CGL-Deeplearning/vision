import unittest
import argparse
import sys
import math
from collections import defaultdict
sys.path.append(".")
import modules.core.CandidateFinder
from modules.core.CandidateFinder import CandidateFinder
from modules.handlers.BamHandler import BamHandler
from modules.handlers.FastaHandler import FastaHandler
from modules.handlers.TextColor import TextColor
from modules.core.IntervalTree import IntervalTree
from modules.handlers.TsvHandler import TsvHandler
from modules.handlers.VcfHandler import VCFFileProcessor
from modules.core.CandidateLabeler import CandidateLabeler
VCF_OFFSET = 1
BAM_FILE = "./unit_test/testdata/NA12878_S1.chr20.10_10p1mb.bam"
FASTA_FILE = "./unit_test/testdata/ucsc.hg19.chr20.unittest.fasta.gz"
VCF_FILE = "./unit_test/testdata/NA12878_S1_chr20.genome.vcf"
REGION_START = 10000000
REGION_STOP = 10010000
CHR_NAME = "chr20"
LOCAL = False

if LOCAL is True:
    BAM_FILE = "/data/users/kishwar/illumina/bam/chr3.bam"
    FASTA_FILE = "/data/users/kishwar/illumina/ref/chr3.fa"
    VCF_FILE = "/data/users/kishwar/illumina/vcf/chr3.vcf"
    CHR_NAME = "chr3"
    REGION_START = 100000
    REGION_STOP = 200000
TYPE_INDEX = {'SNP': 1, 'IN': 2, 'DEL': 3}


class CandidateFinderTest(unittest.TestCase):
    def setUp(self):
        # get the reads that fall in that region
        bam_handler = BamHandler(BAM_FILE)
        fasta_handler = FastaHandler(FASTA_FILE)
        vcf_handler = VCFFileProcessor(file_path=VCF_FILE)
        reads = bam_handler.get_reads(chromosome_name=CHR_NAME,
                                      start=REGION_START,
                                      stop=REGION_STOP)
        # get dictionary of variant records for full region
        vcf_handler.populate_dictionary(contig=None, start_pos=None, end_pos=None, hom_filter=True)

        # get separate positional variant dictionaries for IN, DEL, and SNP
        self.positional_variants = vcf_handler.get_variant_dictionary()
        # create candidate finder object
        candidate_finder = CandidateFinder(reads=reads,
                                           fasta_handler=fasta_handler,
                                           chromosome_name=CHR_NAME,
                                           region_start_position=REGION_START,
                                           region_end_position=REGION_STOP)
        # go through each read and find candidate positions and alleles
        selected_candidates = candidate_finder.parse_reads_and_select_candidates(reads=reads)

        allele_labler = CandidateLabeler(fasta_handler=fasta_handler)
        labeled_sites = allele_labler.get_labeled_candidates(chromosome_name=CHR_NAME,
                                                             positional_vcf=self.positional_variants,
                                                             candidate_sites=selected_candidates)

        self.candidate_dictionary = defaultdict(int)
        for candidate in selected_candidates:
            chr_name, pos_st, pos_end, ref, alt1, alt2, alt1_type, alt2_type, ref_count, alt1_count, alt2_count, \
            alt1_freq, alt2_freq, mq, dp = candidate
            self.candidate_dictionary[pos_st] = (ref, alt1, alt2, mq, dp)

        self.labeled_candidate_dictionary = defaultdict(int)
        for site in labeled_sites:
            chr_name, pos_st, pos_end, ref, alt1, alt2, alt1_type, alt2_type, ref_count, alt1_count, alt2_count, \
            alt1_freq, alt2_freq, mq, dp, gt_alt1, gt_alt2 = site
            self.labeled_candidate_dictionary[pos_st] = (ref, alt1, alt2, gt_alt1, gt_alt2, alt1_type, alt2_type)

    def test_all_position(self):
        for pos in self.positional_variants:
            if pos < REGION_START or pos > REGION_STOP:
                continue
            is_pass = False
            for record_list in self.positional_variants[pos]:
                for record in record_list:
                    if record.filter == "PASS":
                        is_pass = True
            if is_pass is False:
                continue
            bool_val = pos - VCF_OFFSET in self.candidate_dictionary.keys()
            if bool_val is False:
                sys.stderr.write(TextColor.RED + "VCF POSITION NOT PICKED IN CANDIDATE: " + str(pos) + "\n" + TextColor.END)
            self.assertTrue(bool_val)
        sys.stderr.write(TextColor.GREEN + "TEST PASSED: ALL VCF POSITIONS FOUND\n" + TextColor.END)

    def test_refs_found(self):
        for pos in self.positional_variants:
            if pos < REGION_START or pos > REGION_STOP:
                continue
            candidate_ref = self.candidate_dictionary[pos - VCF_OFFSET]
            rec_ref = ''
            for record_list in self.positional_variants[pos]:
                for record in record_list:
                    if record.filter != "PASS":
                        continue
                    rec_ref = record.ref if len(record.ref) > len(rec_ref) else rec_ref
            if rec_ref != '':
                self.assertEqual(rec_ref, candidate_ref[0])
        sys.stderr.write(TextColor.GREEN + "\nTEST PASSED: ALL REFERENCES MATCH\n" + TextColor.END)


    def test_dp(self):
        for pos in self.positional_variants:
            if pos < REGION_START or pos > REGION_STOP:
                continue
            for record_list in self.positional_variants[pos]:
                for record in record_list:
                    if record.filter != "PASS":
                        continue
                    candidate_dp = math.ceil(self.candidate_dictionary[pos - VCF_OFFSET][4])
                    self.assertAlmostEqual(record.dp[0], int(candidate_dp), delta=3)
        sys.stderr.write(TextColor.GREEN + "\nTEST PASSED: ALL COVERAGE MATCH\n" + TextColor.END)

    def test_alts(self):
        for pos in self.positional_variants:
            if pos < REGION_START or pos > REGION_STOP:
                continue
            for record_list in self.positional_variants[pos]:
                for record in record_list:
                    if record.filter != "PASS":
                        continue
                    candidate_ref = self.candidate_dictionary[pos - VCF_OFFSET][1]
                    candidate_alt1 = self.candidate_dictionary[pos - VCF_OFFSET][1]
                    candidate_alt2 = self.candidate_dictionary[pos - VCF_OFFSET][2]
                    found_alt = record.alt == candidate_alt1 or record.alt == candidate_alt2
                    self.assertTrue(found_alt)
        sys.stderr.write(TextColor.GREEN + "\nTEST PASSED: ALL ALTS FOUND\n" + TextColor.END)

    
    def test_labels(self):
        for pos in self.positional_variants:
            if pos < REGION_START or pos > REGION_STOP:
                continue
            for record_list in self.positional_variants[pos]:
                for record in record_list:
                    if record.filter != "PASS":
                        continue
                    candidate_alt1 = self.labeled_candidate_dictionary[pos - VCF_OFFSET][1]
                    candidate_alt2 = self.labeled_candidate_dictionary[pos - VCF_OFFSET][2]
                    candidate_alt1_gt = self.labeled_candidate_dictionary[pos - VCF_OFFSET][3]
                    candidate_alt2_gt = self.labeled_candidate_dictionary[pos - VCF_OFFSET][4]
                    if record.alt == candidate_alt1:
                        vcf_gt = record.gt
                        candidate_gt = candidate_alt1_gt[-1]
                        self.assertListEqual(vcf_gt, candidate_gt)
                    elif record.alt == candidate_alt2:
                        vcf_gt = record.gt
                        candidate_gt = candidate_alt2_gt[-1]
                        self.assertListEqual(vcf_gt, candidate_gt)

        for pos in self.labeled_candidate_dictionary:
            if pos < REGION_START or pos > REGION_STOP:
                continue
            candidate_alt1 = self.labeled_candidate_dictionary[pos][1]
            candidate_alt2 = self.labeled_candidate_dictionary[pos][2]
            candidate_alt1_gt = self.labeled_candidate_dictionary[pos][3]
            candidate_alt2_gt = self.labeled_candidate_dictionary[pos][4]
            candidate_1_found = False if candidate_alt1_gt[0] != 0 else True
            candidate_2_found = False if candidate_alt2_gt[0] != 0 else True
            if candidate_1_found is True and candidate_2_found is True:
                if pos + VCF_OFFSET not in self.positional_variants:
                    continue
            for record_list in self.positional_variants[pos + VCF_OFFSET]:
                for record in record_list:
                    if record.alt == candidate_alt1:
                        vcf_gt = record.gt
                        candidate_gt = candidate_alt1_gt[-1]
                        candidate_1_found = vcf_gt == candidate_gt

                    elif record.alt == candidate_alt2:
                        vcf_gt = record.gt
                        candidate_gt = candidate_alt2_gt[-1]
                        candidate_2_found = vcf_gt == candidate_gt
            if candidate_1_found is False or candidate_2_found is False:
                print(self.labeled_candidate_dictionary[pos])
                for record_list in self.positional_variants[pos + VCF_OFFSET]:
                    for record in record_list:
                        print(record)
            self.assertTrue(candidate_1_found)
            self.assertTrue(candidate_2_found)
        sys.stderr.write(TextColor.GREEN + "\nTEST PASSED: ALL LABELS ARE CORRECT\n" + TextColor.END)


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks to generate the pileup.
    '''
    unittest.main()