import csv
from collections import defaultdict


class SummaryHandler:
    def __init__(self, file_path):
        self.path = file_path
        self.candidate_sites = None

    def get_candidate_positions(self):
        file = open(self.path, 'r')
        reader = csv.reader(file)

        candidate_sites = defaultdict(list)

        for line in reader:
            metadata = line[-1].split()
            chromosome_name, start, stop = metadata[0:3]

            start = int(start)
            stop = int(stop)
            candidate_sites[chromosome_name].append([start, stop])

        for chromosome_name in candidate_sites:
            candidates = candidate_sites[chromosome_name]
            candidate_sites[chromosome_name] = sorted(candidates, key=lambda x: x[0])

        file.close()

        self.candidate_sites = candidate_sites

        return candidate_sites

    def get_chunked_candidate_sites(self, chunk_size):
        if self.candidate_sites is None:
            self.get_candidate_positions()

        chunked_candidate_sites = defaultdict(list)
        for chromosome_name in self.candidate_sites:
            chunk = list()

            region_start_position = self.candidate_sites[chromosome_name][0][0]
            region_end_position = self.candidate_sites[chromosome_name][-1][-1]

            for interval in self.candidate_sites[chromosome_name]:
                start_position, end_position = interval

                chunk.append(interval)

                if end_position - region_start_position >= chunk_size or end_position == region_end_position:
                    chunked_candidate_sites[chromosome_name].append(chunk)
                    chunk = list()
                    region_start_position = end_position

        return chunked_candidate_sites


if __name__ == "__main__":
    file_path = "/home/ryan/data/GIAB/candidate_csv/fl9_realigned_image_labels_chr1_19.csv"
    reader = SummaryHandler(file_path)

    candidate_sites = reader.get_candidate_positions()

