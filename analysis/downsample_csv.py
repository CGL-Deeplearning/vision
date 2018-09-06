import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from collections import OrderedDict
import random

sns.set(color_codes=True)

file_name = sys.argv[1]
downsample_rate = sys.argv[2]
dictionary = dict()
dictionary['0'] = 0
dictionary['1'] = 0
dictionary['2'] = 0
# dictionary['3'] = 0
with open(file_name, "r") as ins:
    for line in ins:
        line = line.rstrip()
        pre_line = line
        if not line:
            continue
        line = line.split(',')
        gt = line[2].split('\t')[-1]
        if downsample_rate < 1.0 and gt == '0':
            random_draw = random.uniform(0, 1)
            if random_draw > downsample_rate:
                continue
        print(pre_line)
        dictionary[gt] += 1

total = dictionary['0'] + dictionary['1'] + dictionary['2']
print("Hom:\t", dictionary['0'], "\t", str((dictionary['0'] * 100) / total), "%")
print("Het:\t", dictionary['1'], "\t", str((dictionary['1'] * 100) / total), "%")
print("Hom-alt:\t", dictionary['2'], "\t", str((dictionary['2'] * 100) / total), "%")
