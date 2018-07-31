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
    def __init__(self, position, adjacent_nodes=None, kmer=None, coverage=1):
        # transition hash keys will be based on sequence only
        self.prev_nodes = set()
        self.next_nodes = set()

        self.positional_coverage = defaultdict(lambda: 0)   # from what position is each read mapping to this node?
        self.coverage = 1                   # assume the node is being created with at least a single read mapped to it
        self.kmer = kmer
        self.coordinate = None
        self.max_coverage_position = None
        self.max_positional_coverage = None

        self.true_variant = False

        self.update_coverage(position)

    def update_coverage(self, position):
        self.coverage += 1
        self.positional_coverage[position] += 1

    def get_max_positional_coverage(self):
        sorted_positional_coverage = sorted(self.positional_coverage.items(), key=lambda x: x[1], reverse=True)

        max_position = sorted_positional_coverage[0][0]
        max_coverage = sorted_positional_coverage[0][1]

        self.max_coverage_position = max_position
        self.max_positional_coverage = max_coverage
        return max_position, max_coverage

    def __str__(self):
        return ' '.join(map(str, [self.kmer]))

    def __hash__(self):
        key = self.kmer
        return hash(key)


class KmerGraph:
    def __init__(self, chromosome_name, start_position, end_position, k, ploidy=2):
        self.chromosome_name = chromosome_name
        self.start_position = start_position
        self.end_position = end_position
        self.length = self.end_position - self.start_position
        self.k = k
        self.ploidy = ploidy

        self.positional_reference = defaultdict(int)
        self.positional_kmers = defaultdict(list)
        self.positional_coverage = defaultdict(int)

        self.node_paths_by_read = dict()
        self.graph = dict()

        # self.max_alleles_per_type = [0, 0, 0, 0]
        self.positional_n_alleles_per_type = dict()

        self.kmer_frequencies = defaultdict(lambda: 0)

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

        kmer = node.kmer

        del self.graph[kmer]

    def initialize_position(self, position):
        template = [{}, {}, {}, {}]
        self.graph[position] = template

    def update_graph(self, read_id, position, kmer, cigar_code, coverage=1):
        if read_id not in self.node_paths_by_read:
            self.node_paths_by_read[read_id] = [None]

        prev_node = self.node_paths_by_read[read_id][-1]

        if kmer not in self.graph:
            node = Node(position=position,
                        kmer=kmer,
                        coverage=coverage)

            self.graph[kmer] = node

        else:
            node = self.graph[kmer]

        # make pointers between nodes
        self.link_nodes(node1=prev_node, node2=node)

        # append node to read path
        self.node_paths_by_read[read_id].append(self.graph[kmer])

        # keep track of which kmers are referred to at each position
        self.positional_kmers[position].append(kmer)

        # update coverage for node
        self.graph[kmer].update_coverage(position)

    def print_graph(self):
        positional_nodes = defaultdict(list)

        for kmer in self.graph:
            node = self.graph[kmer]
            position, coverage = node.get_max_positional_coverage()
            positional_nodes[position].append(node)

        for position in positional_nodes:
            print(position)
            node_strings = list()

            for node in positional_nodes[position]:
                string = "\tKMER:\t" + str(node.kmer) + \
                         "\tMAX_COVERAGE:\t" + str(node.max_positional_coverage)
                node_strings.append(string)

            print('\n'.join(node_strings))

    def plot_graph(self, axes=None, show=True, set_axis_limits=False):
        positional_nodes = defaultdict(list)

        default_color = [0, 0, 0]
        variant_color = [0/255, 153/255, 96/255]

        weights = ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy']

        all_positions = range(self.start_position, self.end_position)
        total_coverage = sum(self.positional_coverage[p] for p in all_positions) / (self.length)

        if axes is None:
            figure = pyplot.figure()
            axes = pyplot.axes()

        min_y = sys.maxsize
        max_y = -sys.maxsize
        min_x = sys.maxsize
        max_x = -sys.maxsize

        for kmer in self.graph:
            node = self.graph[kmer]
            position, coverage = node.get_max_positional_coverage()
            positional_nodes[position].append(node)

        for p, position in enumerate(positional_nodes):
            # sort nodes at this position by their coverage
            # iterate by their coverage and plot from bottom to top

            # find coordinates for all nodes
            for n,node in enumerate(positional_nodes[position]):
                x = p*self.k
                y = n*self.k/2

                node.coordinate = [x,y]

        # finally plot all nodes
        for p, position in enumerate(positional_nodes):
            for n,node in enumerate(positional_nodes[position]):
                kmer = node.kmer
                prev_nodes = node.prev_nodes
                x, y = node.coordinate

                color = default_color
                if node.true_variant:
                    color = variant_color

                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x

                # plot sequence
                weight_index = min(int(round(node.coverage/total_coverage))*5,5)
                weight = weights[weight_index]
                axes.text(x, y, kmer, ha="center", va="center", zorder=2, weight=weight, fontsize=6)

                # plot node shape
                width = (node.coverage/total_coverage)*4
                rectangle_width = self.k/2
                rectangle_height = self.k/4
                p = patches.Rectangle([x-rectangle_width/2, y-rectangle_height/2], width=rectangle_width, height=rectangle_height, zorder=1, facecolor="w", edgecolor=color, alpha=1, linewidth=width)
                axes.add_patch(p)

                # plot edge connecting nodes
                if len(prev_nodes) > 0 and position > self.start_position:
                    for prev_node in prev_nodes:
                        x_prev, y_prev = prev_node.coordinate

                        transition_weight = min([prev_node.coverage, node.coverage])
                        width = min(6*transition_weight/total_coverage, 6)

                        # print(x,y,x_prev,y_prev)

                        p = patches.FancyArrowPatch(posA=[x_prev, y_prev], posB=[x, y], color=default_color, zorder=0,
                                                    lw=width, connectionstyle=patches.ConnectionStyle.Arc3(rad=-0.2))
                        axes.add_patch(p)

                        # if y_prev != y:
                        #     # p = patches.FancyArrowPatch(posA=[x_prev,y_prev], posB=[x,y], zorder=0, lw=width, connectionstyle=patches.ConnectionStyle.Arc3(rad=-0.2))
                        #     # axes.add_patch(p)
                        #     axes.plot([x_prev,x], [y_prev,y], lw=width, color=[0,0,0], zorder=0, alpha=1)
                        #
                        # else:
                        #     axes.plot([x_prev,x], [y_prev,y], lw=width, color=[0,0,0], zorder=0, alpha=1)

        x_limits = [min_x, max_x]
        y_limits = [min_y, max_y]

        if set_axis_limits:
            print(min_x, max_x)
            axes.set_xlim(min_x-self.k, max_x + self.k)
            axes.set_ylim(min_y-self.k, max_y+self.k)
            axes.set_aspect("equal")

        if show:
            pyplot.show()

        return axes, x_limits, y_limits


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
