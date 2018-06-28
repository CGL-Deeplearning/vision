import random
import math
import sys
from collections import defaultdict
from matplotlib import pyplot
from matplotlib import patches


REF_READ_ID = "REFERENCE"

REF = 0
SNP = 1
INS = 2
DEL = 3

def tanh(x):
    e = 1000
    y = (e**x - e**-x) / (e**x + e**-x)
    return y

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

    def __str__(self):
        return ' '.join(map(str, [self.position, self.cigar_code, self.sequence]))

    def __hash__(self):
        key = (self.position, self.cigar_code, self.sequence)
        return hash(key)


class AlignmentGraph:
    def __init__(self, chromosome_name, start_position, end_position, ploidy=2):
        self.chromosome_name = chromosome_name
        self.start_position = start_position
        self.end_position = end_position
        self.length = self.end_position - self.start_position

        self.positional_reference = defaultdict(int)
        self.positional_alleles = defaultdict(int)
        self.positional_coverage = defaultdict(int)
        self.positional_insert_lengths = dict()

        self.node_paths_by_read = dict()

        self.ploidy = ploidy
        self.graph = dict()

        # self.max_alleles_per_type = [0, 0, 0, 0]
        self.positional_n_alleles_per_type = dict()

        self.default_y_position_per_cigar = [0, -1, 2, 1]   # [REF, SNP, INS, DEL]

    def link_nodes(self, node1, node2):
        if node1 is not None and node2 is not None:
            node1.next_nodes.add(node2)
            node2.prev_nodes.add(node1)

    def delete_node(self, node):
        for prev_node in node.prev_nodes:
            prev_node.next_nodes.remove(node)

        for next_node in node.next_nodes:
            next_node.prev_nodes.remove(node)

        position = node.position
        cigar_code = node.cigar_code
        sequence = node.sequence

        del self.graph[position][cigar_code][sequence]

    def update_positional_insert_length(self, position, sequence):
        if position not in self.positional_insert_lengths:
            self.positional_insert_lengths[position] = len(sequence)
        elif len(sequence) > self.positional_insert_lengths[position]:
            self.positional_insert_lengths[position] = len(sequence)

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
            if cigar_code == INS:
                # keep track of inserts for pileup printing purposes
                self.update_positional_insert_length(position, sequence)

            node = Node(position=position,
                        cigar_code=cigar_code,
                        sequence=sequence,
                        coverage=coverage)

            self.graph[position][cigar_code][sequence] = node

        else:
            node = self.graph[position][cigar_code][sequence]
            node.coverage += 1

        # make pointers between nodes
        self.link_nodes(node1=prev_node, node2=node)

        # append node to read path
        self.node_paths_by_read[read_id].append(self.graph[position][cigar_code][sequence])

    def initialize_graph_with_reference(self):
        for position in range(self.start_position, self.end_position+1):
            ref_sequence = self.positional_reference[position]

            self.update_position(read_id=REF_READ_ID, position=position, sequence=ref_sequence, cigar_code=REF)

    def count_alleles_at_position(self, position):
        n_alleles_per_type = [0, 0, 0, 0]

        for cigar_code in [REF, SNP, INS, DEL]:
            for allele in self.graph[position][cigar_code]:
                n_alleles_per_type[cigar_code] += 1

        self.positional_n_alleles_per_type[position] = n_alleles_per_type

    def get_y_position(self, cigar_code, position, n):
        default_position = self.default_y_position_per_cigar[cigar_code]

        if position not in self.positional_n_alleles_per_type:
            self.count_alleles_at_position(position)

        n_alleles_per_type = self.positional_n_alleles_per_type[position]

        offset = 0
        if cigar_code == REF:
            c = 1

        if cigar_code == SNP:
            c = -1

        if cigar_code == INS:
            offset = max(0, n_alleles_per_type[DEL] - 1)
            c = 1

        if cigar_code == DEL:
            c = 1

        y_position = default_position + c*n + offset

        return y_position

    def generate_pileup(self):
        total_inserts = sum([l for l in self.positional_insert_lengths.values()])
        total_length = self.length + total_inserts
        max_coverage = max([c for c in self.positional_coverage.values()])

        pileup = [[None for i in range(total_length)] for j in range(max_coverage)]

        x_offset = 0
        for path in self.node_paths_by_read.values():
            for node in path:
                if node.position < self.start_position or node.position > self.end_position:
                    continue

                # if position in self.positional_insert_lengths:

    def clean_graph(self):
        figure, (axes1, axes2) = pyplot.subplots(nrows=2)

        self.plot_alignment_graph(axes=axes1, show=False)

        for position in range(self.start_position, self.end_position+1):
            for cigar_code in [REF, SNP, DEL, INS]:
                sequences = [key for key in self.graph[position][cigar_code]]

                for s,sequence in enumerate(sequences):
                    node = self.graph[position][cigar_code][sequence]
                    relative_coverage = node.coverage/self.positional_coverage[position]
                    # print(node.coverage, self.positional_coverage[position], relative_coverage, tanh(relative_coverage))

                    if cigar_code == INS and relative_coverage < 0.2:
                        self.delete_node(self.graph[position][cigar_code][sequence])

                    elif cigar_code != DEL and random.random() > tanh(relative_coverage):
                        self.delete_node(self.graph[position][cigar_code][sequence])

        self.plot_alignment_graph(axes=axes2, show=False)

        pyplot.show()

    def print_alignment_graph(self):
        for position in range(self.start_position, self.end_position+1):
            print("REF", [node.sequence for node in self.graph[position][REF].values()],
                  "SNP", [node.sequence for node in self.graph[position][SNP].values()],
                  "DEL", [node.sequence for node in self.graph[position][DEL].values()],
                  "INS", [node.sequence for node in self.graph[position][INS].values()])
        print()

    def plot_alignment_graph(self, axes=None, show=True):
        weights = ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy']

        all_positions = range(self.start_position, self.end_position)
        total_coverage = sum(self.positional_coverage[p] for p in all_positions) / (self.length)

        if axes is None:
            axes = pyplot.axes()

        min_y = sys.maxsize
        max_y = -sys.maxsize

        x_offset = 0
        for position in range(self.start_position, self.end_position+1):
            for cigar_code in [REF, SNP, DEL, INS]:
                nodes = self.graph[position][cigar_code].values()

                if cigar_code == INS and len(nodes) > 0:
                    x_offset += max(1,max([len(seq) for seq in [node.sequence for node in nodes]])/2)

                for n,node in enumerate(nodes):
                    x = position - self.start_position + x_offset
                    y = self.get_y_position(cigar_code, position, n)
                    node.coordinate = [x,y]

                    # plot sequence
                    weight_index = min(int(round(node.coverage/total_coverage))*5,5)
                    weight = weights[weight_index]
                    axes.text(x, y, node.sequence, ha="center", va="center", zorder=2, weight=weight)

                    # plot node shape
                    width = (node.coverage/total_coverage)*5
                    p = patches.Circle([x,y], radius=0.33, zorder=1, facecolor="w", edgecolor="k", alpha=1, linewidth=width)
                    axes.add_patch(p)

                    # plot edge connecting nodes
                    if len(node.prev_nodes) > 0 and position > self.start_position:
                        for prev_node in node.prev_nodes:
                            x_prev, y_prev = prev_node.coordinate
                            transition_weight = min([prev_node.coverage, node.coverage])
                            width = min(6*transition_weight/total_coverage, 6)

                            if y_prev != y:
                                p = patches.FancyArrowPatch(posA=[x_prev,y_prev], posB=[x,y], zorder=0, lw=width, connectionstyle=patches.ConnectionStyle.Arc3(rad=-0.2))
                                axes.add_patch(p)
                            else:
                                axes.plot([x_prev,x], [y_prev,y], lw=width, color=[0,0,0], zorder=0, alpha=1)

                    if y < min_y:
                        min_y = y
                    if y > max_y:
                        max_y = y

        axes.set_xlim(-1, self.length + x_offset + 1)
        axes.set_ylim(min_y-1, max_y+1)
        axes.set_aspect("equal")

        if show:
            pyplot.show()

        return axes
# TO DO:
#   Finish read-only graph
#   Make VCF/REF-based graph
#   Compare read-only graph to VCF/REF-based graph for Nanopore vs Illumina
#   Generate kmer extraction graph method
#       Plot kmer frequency distribution AND label the VCF/REF kmers
#   Generate training data for denoising autoencoder using cleaned-up pileup tensors? Using adjacency??
#   *** Graph-to-pileup method? ***
#   Plot adjacency matrix
#   watershed string idea?
#   *** Naive idea: delete inserts, flip SNPs ***
