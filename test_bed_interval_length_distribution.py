import csv
import numpy
from matplotlib import pyplot


def get_bed_length_distribution():
    bed_file_path = "/home/ryan/data/GIAB/NA12878_GRCh38_confident.bed"

    tsv_file = open(bed_file_path, 'r')
    reader = csv.reader(tsv_file, delimiter='\t')

    lengths = list()
    max_length = 0

    for line in reader:
        # print(line)
        chromosome_name, start, stop = line[0:3]

        start = int(start)
        stop = int(stop)

        length = stop - start

        if length > max_length:
            max_length = length

        lengths.append(length)

    figure = pyplot.figure()
    axes = pyplot.axes()

    print(max_length)

    step = 50
    bins = numpy.arange(0, max_length + step, step=step)
    frequencies, bins = numpy.histogram(lengths, bins=bins)

    frequencies = numpy.log10(frequencies)

    center = (bins[:-1] + bins[1:]) / 2

    axes.bar(center, frequencies, width=step, align="center")

    axes.set_xlabel("Confident Interval Length")
    axes.set_ylabel("log10 frequency")

    pyplot.show()


if __name__ == "__main__":
    get_bed_length_distribution()