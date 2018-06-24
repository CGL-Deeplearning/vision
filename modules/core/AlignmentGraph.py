import random
import math
import sys
from matplotlib import pyplot
from matplotlib import patches


REF_READ_ID = "REFERENCE"

REF = 0
SNP = 1
INS = 2
DEL = 3

class Node:
    def __init__(self, position, cigar_code, prev_node=None, next_node=None, sequence=None, coverage=1):
        # transition keys will be based on position:cigar_code:sequence which is always unique
        self.prev_nodes = set()
        self.next_nodes = set()

        self.position = position
        self.cigar_code = cigar_code
        self.sequence = sequence
        self.coverage = coverage

        self.coordinate = None

        self.add_previous_node(prev_node)
        self.add_next_node(next_node)

    def add_previous_node(self, node):
        if node is not None:
            self.prev_nodes.add(node)

    def add_next_node(self, node):
        if node is not None:
            self.prev_nodes.add(node)

    def __str__(self):
        return ' '.join(map(str, [self.position, self.cigar_code, self.sequence]))

    def __hash__(self):
        key = (self.position, self.cigar_code, self.sequence)
        return hash(key)

class AlignmentGraph:
    def __init__(self, chromosome_name, start_position, end_position, positional_reference=None, positional_alleles=None, positional_coverage=None, ploidy=2):
        self.chromosome_name = chromosome_name
        self.start_position = start_position
        self.end_position = end_position
        self.length = self.end_position - self.start_position

        self.positional_reference = positional_reference
        self.positional_alleles = positional_alleles
        self.positional_coverage = positional_coverage

        self.node_paths_by_read = dict()

        self.ploidy = ploidy
        self.graph = dict()

        # self.max_alleles_per_type = [0, 0, 0, 0]
        self.positional_n_alleles_per_type = dict()

        self.default_y_position_per_cigar = [0, -1, 1, 2]

    def initialize_position(self, position):
        template = [{}, {}, {}, {}]
        self.graph[position] = template

    def update_position(self, read_id, position, sequence, cigar_code, coverage=1):
        if read_id not in self.node_paths_by_read:
            self.node_paths_by_read[read_id] = [None]

        if position not in self.graph:
            self.initialize_position(position=position)

        prev_node = self.node_paths_by_read[read_id][-1]

        if sequence not in self.graph[position][cigar_code]:
            node = Node(position=position,
                        cigar_code=cigar_code,
                        prev_node=prev_node,
                        sequence=sequence,
                        coverage=coverage)

            self.graph[position][cigar_code][sequence] = node

        else:
            self.graph[position][cigar_code][sequence].add_previous_node(prev_node)
            self.graph[position][cigar_code][sequence].coverage += 1

        # append node to read path
        self.node_paths_by_read[read_id].append(self.graph[position][cigar_code][sequence])

    def initialize_graph_with_reference(self):
        for position in range(self.start_position, self.end_position+1):
            ref_sequence = self.positional_reference[position]

            self.update_position(read_id=REF_READ_ID, position=position, sequence=ref_sequence, cigar_code=REF)

    def print_alignment_graph(self):
        for position in range(self.start_position, self.end_position+1):
            print("REF", [node.sequence for node in self.graph[position][REF].values()],
                  "SNP", [node.sequence for node in self.graph[position][SNP].values()],
                  "DEL", [node.sequence for node in self.graph[position][DEL].values()],
                  "INS", [node.sequence for node in self.graph[position][INS].values()])

    def count_alleles_at_position(self, position):
        n_alleles_per_type = [0, 0, 0, 0]

        for cigar_code in [REF, SNP, INS, DEL]:
            for allele in self.graph[position][cigar_code]:
                n_alleles_per_type[cigar_code] += 1

        self.positional_n_alleles_per_type[position] = n_alleles_per_type

    def get_y_position(self, cigar_code, position, n):
        # if REF, default pos = 0
        # if SNP, default pos also = 0
        # if INS, default pos = -1
        # if DEL, default pos = 1

        default_position = self.default_y_position_per_cigar[cigar_code]

        if position not in self.positional_n_alleles_per_type:
            self.count_alleles_at_position(position)

        n_alleles_per_type = self.positional_n_alleles_per_type[position]

        height = n_alleles_per_type[cigar_code]

        y_position = default_position + n - (height-1)

        return y_position

    def plot_alignment_graph(self):
        figure = pyplot.figure()
        axes = pyplot.axes()

        min_y = sys.maxsize
        max_y = -sys.maxsize

        x_offset = 0
        for position in range(self.start_position, self.end_position+1):
            for cigar_code in [REF, SNP, INS, DEL]:

                nodes = self.graph[position][cigar_code].values()
                if cigar_code == INS and len(nodes) > 0:
                    x_offset += max([len(seq) for seq in [node.sequence for node in nodes]]) - 1

                for n,node in enumerate(nodes):
                    x = position - self.start_position + x_offset
                    y = self.get_y_position(cigar_code, position, n)
                    node.coordinate = [x,y]

                    # plot sequence
                    pyplot.text(x, y, node.sequence, ha="center", va="center", zorder=1)

                    # plot node shape
                    p = patches.Circle([x,y], radius=0.33, zorder=0, facecolor="w", edgecolor="k")
                    axes.add_patch(p)

                    # plot edge connecting nodes
                    if len(node.prev_nodes) > 0 and position > self.start_position:
                        for prev_node in node.prev_nodes:
                            x_prev, y_prev = prev_node.coordinate
                            coverage = min([prev_node.coverage, node.coverage])
                            width = min(4*coverage/30, 4)
                            pyplot.plot([x_prev,x], [y_prev,y], lw=width, color="k", zorder=-1)

                    if y < min_y:
                        min_y = y
                    if y > max_y:
                        max_y = y

        axes.set_xlim(-1, self.length + x_offset + 1)
        axes.set_ylim(min_y-1, max_y+1)
        axes.set_aspect("equal")

        pyplot.show()

# TO DO:
#   Finish read-only graph
#       Make plotting method for graph
#   Make VCF/REF-based graph
#   Compare read-only graph to VCF/REF-based graph for Nanopore vs Illumina
#   Generate kmer extraction graph method
#       Plot kmer frequency distribution AND label the VCF/REF kmers
#   *** Generate heuristic for cleanup ***
#   Generate training data for denoising autoencoder using cleaned-up pileup tensors
#   Graph-to-pileup method?