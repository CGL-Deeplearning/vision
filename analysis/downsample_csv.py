import sys
import random
from collections import defaultdict
HOM = 0
HET = 1
HOM_ALT = 2

file_name = sys.argv[1]
downsample_rate = float(sys.argv[2])

dictionary = dict()
dictionary[HOM] = 0
dictionary[HET] = 0
dictionary[HOM_ALT] = 0

records = defaultdict(list)

with open(file_name, "r") as ins:
    for line in ins:
        line = line.rstrip()
        pre_line = line
        if not line:
            continue
        line = line.split(',')
        gt = int(line[2].split('\t')[-1])
        if downsample_rate < 1.0 and gt == HOM:
            random_draw = random.uniform(0, 1)
            if random_draw > downsample_rate:
                continue

        dictionary[gt] += 1
        records[gt].append(pre_line)

sys.stderr.write("After down-sampling:\n")
total = dictionary[HOM] + dictionary[HET] + dictionary[HOM_ALT]
sys.stderr.write(str("Hom:\t\t" + str(dictionary[HOM]) + "\t" + str((dictionary[HOM] * 100) / total) + "%" + "\n"))
sys.stderr.write(str("Het:\t\t" + str(dictionary[HET]) + "\t" + str((dictionary[HET] * 100) / total) + "%" + "\n"))
sys.stderr.write(str("Hom-alt:\t" + str(dictionary[HOM_ALT]) + "\t" + str((dictionary[HOM_ALT] * 100) / total) + "%" + "\n"))

het_over_sampling_ratio = (float(dictionary[HOM]) / float(dictionary[HET] * 3.0)) if dictionary[HET] else 0
hom_alt_over_sampling_ratio = (float(dictionary[HOM]) / float(dictionary[HOM_ALT] * 3.0)) if dictionary[HOM_ALT] else 0

if het_over_sampling_ratio > 1:
    records[HET] = list(records[HET]) * int(het_over_sampling_ratio - 1)
if hom_alt_over_sampling_ratio > 1:
    records[HOM_ALT] = list(records[HOM_ALT]) * int(hom_alt_over_sampling_ratio - 1)

dictionary = dict()
dictionary[HOM] = 0
dictionary[HET] = 0
dictionary[HOM_ALT] = 0
all_records = records[HOM] + records[HET] + records[HOM_ALT]
random.shuffle(all_records)

for line in all_records:
    pre_line = line
    line = line.split(',')
    gt = int(line[2].split('\t')[-1])
    dictionary[gt] += 1
    print(pre_line)

sys.stderr.write("After over-sampling:\n")
total = dictionary[HOM] + dictionary[HET] + dictionary[HOM_ALT]
sys.stderr.write(str("Hom:\t\t" + str(dictionary[HOM]) + "\t" + str((dictionary[HOM] * 100) / total) + "%" + "\n"))
sys.stderr.write(str("Hom:\t\t" + str(dictionary[HET]) + "\t" + str((dictionary[HET] * 100) / total) + "%" + "\n"))
sys.stderr.write(str("Hom-alt:\t" + str(dictionary[HOM_ALT]) + "\t" + str((dictionary[HOM_ALT] * 100) / total) + "%" + "\n"))