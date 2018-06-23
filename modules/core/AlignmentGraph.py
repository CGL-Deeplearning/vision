import random
import math
from matplotlib import pyplot


REF = 0
SNP = 1
INS = 2
DEL = 3


class Node:
    def __init__(self, prev_node=None, next_node=None, sequence=None, coverage=1):
        self.prev_node = prev_node
        self.next_node = next_node

        self.sequence = sequence
        self.coverage = coverage


class AlignmentGraph:
    def __init__(self, chromosome_name, start_position, end_position, positional_reference, positional_alleles=None, positional_coverage=None, ploidy=2):
        self.chromosome_name = chromosome_name
        self.start_position = start_position
        self.end_position = end_position
        self.length = self.end_position - self.start_position

        self.positional_reference = positional_reference
        self.positional_alleles = positional_alleles
        self.positional_coverage = positional_coverage

        self.ploidy = ploidy
        self.graph = dict()

        self.max_alleles_per_type = [0, 0, 0, 0]
        self.positional_n_alleles_per_type = dict()

        self.default_y_position_per_cigar = [0, 0, -1, 1]

    def initialize_position(self, position):
        template = [{}, {}, {}, {}]
        self.graph[position] = template

    def update_position(self, position, sequence, cigar_code, prev_node=None, coverage=1):
        if position not in self.graph:
            self.initialize_position(position=position)

        if sequence not in self.graph[position][cigar_code]:
            node = Node(prev_node=prev_node, sequence=sequence, coverage=coverage)
            self.graph[position][cigar_code][sequence] = node
        else:
            self.graph[position][cigar_code][sequence].coverage += 1

    def initialize_graph_with_reference(self):
        for position in range(self.start_position, self.end_position+1):
            ref_sequence = self.positional_reference[position]

            if position == self.start_position:
                prev_node = None
            else:
                prev_node = self.graph[position-1][REF]

            self.update_position(position=position, sequence=ref_sequence, cigar_code=REF, prev_node=prev_node)

    def parse_positional_alleles(self, position, prev_node):
        n_alleles_per_type = [0, 0, 0, 0]
        n_alleles_per_type[REF] = int(self.positional_coverage[position] - sum(n_alleles_per_type) > 0)

        for allele in self.positional_alleles[position].items():
            allele_tuple, coverage = allele
            sequence, cigar_code = allele_tuple

            self.update_position(position=position,
                                 sequence=sequence,
                                 cigar_code=cigar_code,
                                 prev_node=prev_node,
                                 coverage=coverage)

            n_alleles_per_type[cigar_code] += 1

        # probably not worth doing this... just let matplotlib find the limits
        self.max_alleles_per_type = [max(n1, n2) for n1,n2 in zip(self.max_alleles_per_type, n_alleles_per_type)]
        self.positional_n_alleles_per_type[position] = n_alleles_per_type

    def parse_positional_reference(self, position, prev_node):
        coverage = self.positional_coverage[position]
        ref_sequence = self.positional_reference[position]

        self.update_position(position=position,
                             sequence=ref_sequence,
                             cigar_code=REF,
                             prev_node=prev_node,
                             coverage=coverage)

        self.positional_n_alleles_per_type[position] = [1, 0, 0, 0]

    def generate_graph_from_positional_data(self):
        for position in range(self.start_position, self.end_position+1):
            # Initialize the position if it does not exist yet
            if position == self.start_position:
                prev_node = None
            else:
                prev_node = self.graph[position-1][REF]

            # Update the graph depending on whether there are alternate alleles found at this position
            if position in self.positional_alleles:
                # there is some non-reference alignment here
                self.parse_positional_alleles(position=position, prev_node=prev_node)

            else:
                # there are only reference-matching reads here
                self.parse_positional_reference(position=position, prev_node=prev_node)

    def print_alignment_graph(self):
        for position in range(self.start_position, self.end_position+1):
            print("REF", [node.sequence for node in self.graph[position][REF].values()],
                  "SNP", [node.sequence for node in self.graph[position][SNP].values()],
                  "DEL", [node.sequence for node in self.graph[position][DEL].values()],
                  "INS", [node.sequence for node in self.graph[position][INS].values()])

    def get_y_position(self, cigar_code, position, n):
        n_alleles_per_type = self.positional_n_alleles_per_type[position]

        # if REF, default pos = 0
        # if SNP, default pos also = 0
        # if INS, default pos = -1
        # if DEL, default pos = 1

        default_position = self.default_y_position_per_cigar[cigar_code]
        height = n_alleles_per_type[cigar_code]

        y_position = default_position + n/height - sum(n_alleles_per_type[:cigar_code])/2

        return y_position

    def plot_alignment_graph(self):
        figure = pyplot.figure()
        axes = pyplot.axes()

        height = sum(self.max_alleles_per_type)

        axes.set_xlim(-1, self.length+1)
        axes.set_ylim(-height/2, height/2)

        for position in range(self.start_position, self.end_position+1):
            for cigar_code in [REF, SNP, INS, DEL]:
                print(cigar_code)
                print(self.graph[position][cigar_code].values())
                for n,node in enumerate(self.graph[position][cigar_code].values()):
                    x = position - self.start_position
                    y = self.get_y_position(cigar_code, position, n)

                    pyplot.text(x,y,node.sequence,ha="center",va="center")
                    print(node.sequence)


        pyplot.show()

# TO DO:
#   Finish read-only graph
#       Make plotting method for graph
#   Make VCF/REF-based graph
#   Compare read-only graph to VCF/REF-based graph for Nanopore vs Illumina
#   Generate kmer extraction graph method
#       Plot kmer frequency distribution AND label the VCF/REF kmers
#   If success, convert to one-pass method by updating graph directly through candidate_finder
#   ***Generate heuristic for cleanup***
#   generate training data for denoising autoencoder using cleaned-up pileup tensors