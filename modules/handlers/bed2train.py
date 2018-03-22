import sys
import random
from modules.handlers.TextColor import TextColor

CLASS_BY_INDEX = ["HOM", "HET", "HOM_ALT"]

class_count = {}


def get_combined_gt(gt1, gt2):
    if gt1 == '0':
        return gt2
    if gt2 == '0':
        return gt1
    if gt1 == '0' and gt2 == '0':
        return '0'
    if gt1 == '1' and gt2 == '1':
        return '2'
    return None


def _initialize_class_count_dictionary(chr_name):
    if chr_name not in class_count.keys():
        class_count[chr_name] = {'0': 0, '1': 0, '2': 0}


def get_images_for_two_alts(record):
    chr_name, pos_start, pos_end, ref, alt1, alt2, rec_type = record.rstrip().split('\t')[0:7]
    gt1, gt2 = record.rstrip().split('\t')[-2:]
    gt1 = gt1[1]
    gt2 = gt2[1]
    _initialize_class_count_dictionary(chr_name)

    class_count[chr_name][gt1] += 1
    class_count[chr_name][gt2] += 1
    gt3 = get_combined_gt(gt1, gt2)
    if gt3 is None:
        sys.stderr.write(TextColor.RED + "WEIRD RECORD: " + str(record) + "\n")
    rec_1 = [chr_name, pos_start, pos_end, ref, alt1, '.', rec_type, gt1]
    rec_2 = [chr_name, pos_start, pos_end, ref, alt2, '.', rec_type, gt2]
    if gt3 is not None:
        class_count[chr_name][str(gt3)] += 1
        rec_3 = [chr_name, pos_start, pos_end, ref, alt1, alt2, rec_type, gt3]
        return [rec_1, rec_2, rec_3]

    return [rec_1, rec_2]


def select_or_not(downsample_rate):
    """
    Determines if a bed record should be selected given a downsampling rate
    :param bed_record: A bed record
    :param downsample_rate: A downsampling probability
    :return: Boolean
    """
    # else do a sampling based on probability
    random_chance = random.uniform(0, 1)
    if random_chance <= downsample_rate:
        return True
    return False


def get_downsample_rate(total_hom, total_het, total_hom_alt):
    """
    Downsample the bed file
    :return:
    """
    # calculate the downsample rate based on distribution of three classes
    downsample_rate = max(total_het, total_hom_alt) / total_hom if total_hom else 0
    # we want the homozygous to be twice the size of the next most frequent class.
    downsample_rate = 2 * downsample_rate

    return downsample_rate


def print_class_distribution(train_set, class_count):
    hom, het, hom_alt = 0, 0, 0
    for chr_name in train_set:
        hom += class_count[chr_name]['0']
        het += class_count[chr_name]['1']
        hom_alt += class_count[chr_name]['2']
    m_fac = 100/(hom + het + hom_alt) if hom + het + hom_alt else 0
    sys.stderr.write(TextColor.GREEN + "Total hom records:\t\t" + str(hom) + "\t" + str(hom * m_fac) + "\n")
    sys.stderr.write("Total het records:\t\t" + str(het) + "\t" + str(het * m_fac) + "\n")
    sys.stderr.write("Total hom_alt records:\t\t" + str(hom_alt) + "\t" + str(hom_alt * m_fac) + "\n" + TextColor.END)


def get_prediction_set_from_bed(candidate_bed):
    with open(candidate_bed) as bed_file:
        bed_records = bed_file.readlines()

    train_set = {}

    rec_id = 1
    for record in bed_records:

        chr_name, pos_start, pos_end, ref, alt1, alt2, rec_type = record.rstrip().split('\t')[0:7]
        gt1, gt2 = record.rstrip().split('\t')[-2:]
        gt1 = gt1[1]

        if chr_name not in train_set.keys():
            train_set[chr_name] = []
        _initialize_class_count_dictionary(chr_name)

        if alt2 != '.':
            train_set[chr_name].extend(get_images_for_two_alts(record))
        else:
            train_set[chr_name].append([chr_name, pos_start, pos_end, ref, alt1, alt2, rec_type, gt1])
            class_count[chr_name][gt1] += 1
        rec_id += 1

    sys.stderr.write(TextColor.BLUE + "Raw class distribution\n"+TextColor.END)
    print_class_distribution(train_set, class_count)

    for chr_name in train_set:
        downsample_rate = get_downsample_rate(class_count[chr_name]['0'], class_count[chr_name]['1'], class_count[chr_name]['2'])
        if downsample_rate >= 1:
            continue
        downsampled_list = []
        for prediction in train_set[chr_name]:
            alt2 = prediction[5]
            gt = prediction[7]
            if gt == '0' and alt2 == '.' and select_or_not(downsample_rate) is False:
                continue
            downsampled_list.append(prediction)
        train_set[chr_name] = downsampled_list

    sys.stderr.write(TextColor.BLUE + "Downsampled class distribution\n" + TextColor.END)
    print_class_distribution(train_set, class_count)

    return train_set


def get_flattened_bed(train_set):
    flattened_list = []
    for chr_name in train_set:
        for record in train_set[chr_name]:
            flattened_list.append(record)
    return flattened_list