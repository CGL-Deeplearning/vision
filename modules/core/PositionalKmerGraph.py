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

        self.coverage = 0
        self.kmer = kmer
        self.coordinate = None
        self.position = position

    def __str__(self):
        return ' '.join(map(str, [self.position, self.kmer]))

    def __hash__(self):
        key = self.kmer
        return hash(key)


class PositionalKmerGraph:
    def __init__(self, chromosome_name, start_position, end_position, k, ploidy=2):
        self.chromosome_name = chromosome_name
        self.start_position = start_position
        self.end_position = end_position
        self.length = self.end_position - self.start_position
        self.k = k

        self.positional_reference = defaultdict(int)
        self.positional_kmers = defaultdict(list)
        self.positional_coverage = defaultdict(lambda: 0)

        self.node_paths_by_read = dict()

        self.ploidy = ploidy
        self.graph = defaultdict(dict)

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

        kmer = node.kmer
        position = node.position

        del self.graph[position][kmer]

    def initialize_position(self, position):
        template = [{}, {}, {}, {}]
        self.graph[position] = template

    def update_graph(self, read_id, position, kmer, cigar_code, coverage=1):
        if read_id not in self.node_paths_by_read:
            self.node_paths_by_read[read_id] = [None]

        prev_node = self.node_paths_by_read[read_id][-1]

        if kmer not in self.graph[position]:
            node = Node(position=position,
                        kmer=kmer,
                        coverage=coverage)

            self.graph[position][kmer] = node

        else:
            node = self.graph[position][kmer]

        # make pointers between nodes
        self.link_nodes(node1=prev_node, node2=node)

        # append node to read path
        self.node_paths_by_read[read_id].append(self.graph[position][kmer])

        # keep track of which kmers are referred to at each position
        self.positional_kmers[position].append(kmer)

        # update coverage for node
        self.graph[position][kmer].coverage += 1

    def print_graph(self):
        positional_nodes = defaultdict(list)

        for position in sorted(self.graph):
            print(position)

            for kmer in self.graph[position]:
                node = self.graph[position][kmer]

                node_strings = list()

                string = "\tKMER:\t" + str(node.kmer) + \
                         "\tMAX_COVERAGE:\t" + str(node.coverage)
                node_strings.append(string)

                print('\n'.join(node_strings))

    def plot_graph(self, axes=None, show=True):
        positional_nodes = defaultdict(list)

        weights = ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy']

        all_positions = range(self.start_position, self.end_position)
        total_coverage = sum(self.positional_coverage[p] for p in all_positions) / (self.length)

        # print(list(self.positional_coverage.items()))

        if axes is None:
            figure = pyplot.figure()
            figure.set_size_inches(w=14,h=4)
            axes = pyplot.axes()

        min_y = sys.maxsize
        max_y = -sys.maxsize
        min_x = sys.maxsize
        max_x = -sys.maxsize

        # for p, position in enumerate(sorted(self.graph)):
        #     nodes = self.graph[position].values()
        #     # nodes = sorted(nodes, key=lambda x: x.coverage, reverse=True)
        #     for n, node in enumerate(nodes):
        #         x = p * self.k
        #         y = n
        #
        #         node.coordinate = [x,y]

        for p, position in enumerate(sorted(self.graph)):
            nodes = self.graph[position].values()
            # nodes = sorted(nodes, key=lambda x: x.coverage, reverse=True)
            for n, node in enumerate(nodes):
                kmer = node.kmer

                # total_coverage = self.positional_coverage[position] + 1
                # print(total_coverage)

                prev_nodes = node.prev_nodes

                # x,y = node.coordinate
                x = p * self.k
                y = n

                if node.coordinate is None:
                    self.graph[position][kmer].coordinate = [x,y]

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
                # axes.text(x, y, kmer, ha="center", va="center", zorder=2, weight=weight)

                # plot node shape
                width = (node.coverage/total_coverage)*5
                patch = patches.Circle([x,y], radius=0.33, zorder=1, facecolor="w", edgecolor="k", alpha=1, linewidth=width)
                axes.add_patch(patch)

                # plot edge connecting nodes
                if len(prev_nodes) > 0 and position > self.start_position:
                    for prev_node in prev_nodes:
                        # print("prev", prev_node)

                        # if position - prev_node.position > 2:
                        #     continue
                        #
                        # if prev_node.coordinate is None:
                        #     continue

                        x_prev, y_prev = prev_node.coordinate

                        transition_weight = min([prev_node.coverage, node.coverage])
                        width = min(6*transition_weight/total_coverage, 6)

                        # print(x,y,x_prev,y_prev)

                        if y_prev != y:
                            patch = patches.FancyArrowPatch(posA=[x_prev,y_prev], posB=[x,y], zorder=0, lw=width, connectionstyle=patches.ConnectionStyle.Arc3(rad=-0.2))
                            axes.add_patch(patch)
                        else:
                            axes.plot([x_prev,x], [y_prev,y], lw=width, color=[0,0,0], zorder=0, alpha=1)

        # print(min_x, max_x)
        axes.set_xlim(min_x-self.k, max_x + self.k)
        axes.set_ylim(min_y-self.k, max_y+self.k)
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
