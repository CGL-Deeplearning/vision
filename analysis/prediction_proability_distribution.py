import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import argparse
import seaborn as sns
from collections import defaultdict
import numpy as np

sns.set(color_codes=True)
MULTI_ALLELE_QUAL_FILTER = 5.0
SINGLE_ALLELE_SNP_QUAL_FILTER = 3.0
SINGLE_ALLELE_INDEL_QUAL_FILTER = 5.0


def read_file(file_name):
    positional_records = defaultdict(list)
    # gather all positional records
    with open(file_name, "r") as ins:
        counter = 0
        for line in ins:
            line = line.rstrip()
            if not line:
                continue
            line = line.split('\t')
            true_label, predicted_label, record, probs = line[1], line[2], line[3:12], line[12:]
            chromosome_name, position_start, posistion_end, ref, alt1, alt2, alt1_type, alt2_type, genotype = record
            positional_records[position_start].append(line)
            # if counter == 100000:
            #     break
            counter += 1

    # split records to single allelelic and multi-alleleic
    single_allele_records = defaultdict(list)
    multi_allele_records = defaultdict(list)
    total_multi_allelic_positions = 0
    total_single_allelic_positions = 0
    total_positions = 0
    for pos in positional_records:
        total_positions += 1
        if len(positional_records[pos]) == 3:
            for record in positional_records[pos]:
                multi_allele_records[pos].append(record)
            total_multi_allelic_positions += 1
        elif len(positional_records[pos]) == 1:
            single_allele_records[pos].append(positional_records[pos][0])
            total_single_allelic_positions += 1
        else:
            print("POSITION HAS INCONSISTENT NUMBER OF RECORDS")
            print(positional_records[pos])

    sys.stderr.write("TOTAL POSITIONS THAT HAVE ALLELES: " + str(total_positions) + "\n")
    sys.stderr.write("TOTAL POSITIONS WITH SNGL ALLELES: " + str(total_single_allelic_positions) + " "
                     + str(total_single_allelic_positions * 100 / total_positions) + "%\n")
    sys.stderr.write("TOTAL POSITIONS WITH MULT ALLELES: " + str(total_multi_allelic_positions) + " "
                     + str(total_multi_allelic_positions * 100 / total_positions) + "%\n")
    print("-------------------------------")

    return single_allele_records, multi_allele_records


def get_genotype_for_single_allele(probabilities, true_gt_index, alt_type):
    genotype_list = ['0/0', '0/1', '1/1']
    qual = sum(probabilities) - probabilities[0]
    phred_qual = min(60, -10 * np.log10(1 - qual) if 1 - qual >= 0.0000001 else 60)

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

    return genotype_list[index], phred_qual, phred_gq, genotype_list[true_gt_index]


def parse_single_allelic_records(single_allele_records):
    # SINGLE ALLELE PREDICTION DISTRIBUTION
    right_snp_calls = []
    right_indel_calls = []
    wrong_snp_calls = []
    wrong_indel_calls = []
    fp_snp = 0
    fp_indel = 0
    fn_snp = 0
    fn_indel = 0
    for pos in single_allele_records:
        line = single_allele_records[pos][0]
        chromosome_name, position_start, posistion_end, ref, alt1, alt2, alt1_type, alt2_type, genotype = line[3:12]
        preds = [float(line[12]), float(line[13]), float(line[14])]
        predicted_genotype, phred_qual, phred_gq, true_genotype = \
            get_genotype_for_single_allele(preds, int(line[1]), int(alt1_type))

        if true_genotype == '0/0' and predicted_genotype == '0/0':
            continue
        if true_genotype == predicted_genotype and int(alt1_type) == 1:
            right_snp_calls.append(phred_qual)
        elif true_genotype == predicted_genotype:
            right_indel_calls.append(phred_qual)

        elif true_genotype != predicted_genotype and int(alt1_type) == 1:
            wrong_snp_calls.append(phred_qual)
            if true_genotype != '0/0' and predicted_genotype == '0/0':
                fn_snp += 1
            else:
                fp_snp += 1
        elif true_genotype != predicted_genotype:
            wrong_indel_calls.append(phred_qual)
            if true_genotype != '0/0' and predicted_genotype == '0/0':
                fn_indel += 1
            else:
                fp_indel += 1

    print("SINGLE ALLELE SNP ACCURACY:")
    print("FALSE POSITIVE:\t", str(fp_snp))
    print("FALSE NEGATIVE:\t", str(fn_snp))
    print("-------------------------------")
    print("SINGLE ALLELE INDEL ACCURACY:")
    print("FALSE POSITIVE:\t", str(fp_indel))
    print("FALSE NEGATIVE:\t", str(fn_indel))
    print("-------------------------------")

    total_fp = fp_snp + fp_indel
    total_fn = fn_snp + fn_indel

    return right_snp_calls, right_indel_calls, wrong_snp_calls, wrong_indel_calls, total_fp, total_fn


def get_true_genotype_for_multiple_alleles(rec_alt1_label, rec_alt2_label, combined_label):
    genotype_list = ['0/0', '0/1', '1/1', '0/2', '2/2', '1/2']
    if rec_alt1_label == 1 and rec_alt2_label == 1 and combined_label == 2:
        return '1/2'
    if rec_alt1_label == 1 and rec_alt2_label == 0 and combined_label == 1:
        return '0/1'
    if rec_alt1_label == 0 and rec_alt2_label == 1 and combined_label == 1:
        return '0/2'
    if rec_alt1_label == 2 and rec_alt2_label == 0 and combined_label == 2:
        return '1/1'
    if rec_alt1_label == 0 and rec_alt2_label == 2 and combined_label == 2:
        return '2/2'
    if rec_alt1_label == 0 and rec_alt2_label == 0 and combined_label == 0:
        return '0/0'
    return None


def get_predicted_genotype_for_multiple_allele(records):
    ref = '.'
    st_pos = 0
    end_pos = 0
    chrm = ''
    rec_alt1 = '.'
    rec_alt2 = '.'
    alt_probs = defaultdict(tuple)
    true_genotype = defaultdict(int)
    for record in records:
        chrm = record[3]
        st_pos = record[4]
        end_pos = record[5]
        ref = record[6]
        alt1 = record[7]
        alt2 = record[8]
        alt1_type = record[9]
        alt2_type = record[10]
        label = record[11]
        probs = (float(record[12]), float(record[13]), float(record[14]))
        if alt1 != '.' and alt2 != '.':
            rec_alt1 = (alt1, alt1_type)
            rec_alt2 = (alt2, alt2_type)
            alt_probs['both'] = probs
            true_genotype['both'] = int(label)
        else:
            true_genotype[(alt1, alt1_type)] = int(label)
            alt_probs[(alt1, alt1_type)] = probs
    p00 = min(alt_probs[rec_alt1][0], alt_probs[rec_alt2][0], alt_probs['both'][0])
    p01 = min(alt_probs[rec_alt1][1], alt_probs['both'][1])
    p11 = min(alt_probs[rec_alt1][2], alt_probs['both'][2])
    p02 = min(alt_probs[rec_alt2][1], alt_probs['both'][1])
    p22 = min(alt_probs[rec_alt2][2], alt_probs['both'][2])
    p12 = alt_probs['both'][2]

    true_gt = get_true_genotype_for_multiple_alleles(true_genotype[rec_alt1], true_genotype[rec_alt2], true_genotype['both'])
    if true_gt is None:
        print("INVALID RECORDS IN POSITION: ", st_pos)
        print(records)
        exit()
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
            return chrm, st_pos, end_pos, ref, [rec_alt1, rec_alt2], genotype_list[index], phred_qual, phred_gq, true_gt
        else:
            probs = [p00, p02, p22]
            sum_probs = sum(probs)
            probs = [(float(i) / sum_probs) if sum_probs else 0 for i in probs]

            genotype_list = ['0/0', '0/2', '2/2']
            gq, index = max([(v, i) for i, v in enumerate(probs)])
            qual = sum(probs) - probs[0]
            phred_qual = min(60, -10 * np.log10(1 - qual) if 1 - qual >= 0.0000001 else 60)
            phred_gq = min(60, -10 * np.log10(1 - gq) if 1 - gq >= 0.0000001 else 60)
            return chrm, st_pos, end_pos, ref, [rec_alt1, rec_alt2], genotype_list[index], phred_qual, phred_gq, true_gt
    elif alt1_phred_qual < MULTI_ALLELE_QUAL_FILTER:
        probs = [p00, p02, p22]
        sum_probs = sum(probs)
        probs = [(float(i) / sum_probs) if sum_probs else 0 for i in probs]

        genotype_list = ['0/0', '0/2', '2/2']
        gq, index = max([(v, i) for i, v in enumerate(probs)])
        qual = sum(probs) - probs[0]
        phred_qual = min(60, -10 * np.log10(1 - qual) if 1 - qual >= 0.0000001 else 60)
        phred_gq = min(60, -10 * np.log10(1 - gq) if 1 - gq >= 0.0000001 else 60)
        return chrm, st_pos, end_pos, ref, [rec_alt1, rec_alt2], genotype_list[index], phred_qual, phred_gq, true_gt
    elif alt2_phred_qual < MULTI_ALLELE_QUAL_FILTER:
        probs = [p00, p01, p11]
        sum_probs = sum(probs)
        probs = [(float(i) / sum_probs) if sum_probs else 0 for i in probs]

        genotype_list = ['0/0', '0/1', '1/1']
        gq, index = max([(v, i) for i, v in enumerate(probs)])
        qual = sum(probs) - probs[0]
        phred_qual = min(60, -10 * np.log10(1 - qual) if 1 - qual >= 0.0000001 else 60)
        phred_gq = min(60, -10 * np.log10(1 - gq) if 1 - gq >= 0.0000001 else 60)
        return chrm, st_pos, end_pos, ref, [rec_alt1, rec_alt2], genotype_list[index], phred_qual, phred_gq, true_gt
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
        return chrm, st_pos, end_pos, ref, [rec_alt1, rec_alt2], genotype_list[index], phred_qual, phred_gq, true_gt


def parse_multi_allelic_records(multi_allele_records):
    # SINGLE ALLELE PREDICTION DISTRIBUTION
    right_calls_qual = []
    right_calls_gq = []
    wrong_calls_qual = []
    wrong_calls_gq = []
    fn = 0
    fp = 0
    for pos in multi_allele_records:
        chrm, st_pos, end_pos, ref, alts, pred_gt, phred_qual, phred_gq, true_gt = \
            get_predicted_genotype_for_multiple_allele(multi_allele_records[pos])
        if true_gt == '0/0' and pred_gt == '0/0':
            continue
        if pred_gt == true_gt:
            right_calls_qual.append(phred_qual)
            right_calls_gq.append(phred_gq)
        else:
            wrong_calls_qual.append(phred_qual)
            wrong_calls_gq.append(phred_gq)
            if true_gt != '0/0' and pred_gt == '0/0':
                fn += 1
            else:
                fp += 1

    print("MULTI ALLELE ACCURACY:")
    print("FALSE POSITIVE:\t", str(fp))
    print("FALSE NEGATIVE:\t", str(fn))
    print("-------------------------------")

    return right_calls_qual, right_calls_gq, wrong_calls_qual, wrong_calls_gq, fp, fn


def plot_historgram(axes, right_predictions, wrong_predictins, title_string):
    axes.hist(right_predictions, bins=60, alpha=0.5, label='right predictions', color='green', normed=1)
    axes.hist(wrong_predictins, bins=60, alpha=0.5, label='wrong predictions', color='red', normed=1)
    axes.set_xlabel("Phred scores")
    axes.set_ylabel("Frequency")
    axes.set_title(title_string)
    axes.legend()


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="CSV generated through test.py."
    )
    FLAGS, unparsed = parser.parse_known_args()
    single_allelic_records, multi_allelic_recods = read_file(FLAGS.csv_file)

    right_snp, right_indel, wrong_snp, wrong_indel, false_positives_sngl, false_negative_sngl \
        = parse_single_allelic_records(single_allelic_records)

    right_qual, right_gq, wrong_qual, wrong_gq, false_positives_mult, false_negative_mult = \
        parse_multi_allelic_records(multi_allelic_recods)

    overall_fp = false_positives_sngl + false_positives_mult
    overall_fn = false_negative_sngl + false_negative_sngl
    print("Overall ACCURACY:")
    print("FALSE POSITIVE:\t", str(overall_fp))
    print("FALSE NEGATIVE:\t", str(overall_fn))
    print("-------------------------------")
    # plt.ylim((0, 200))
    # fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(16, 8), sharey=False, frameon=True)
    # plot_historgram(axes[0, 0], right_snp, wrong_snp, "Single alleleic site - SNP")
    # plot_historgram(axes[0, 1], right_indel, wrong_indel, "Single allelic site - INDEL")
    # plot_historgram(axes[1, 0], right_qual, wrong_qual, "MULTI alleleic site - QUAL")
    # plot_historgram(axes[1, 1], right_gq, wrong_gq, "MULTI allelic site - GQ")
    # plt.show()
