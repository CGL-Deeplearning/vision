import random
import math
import sys
from collections import defaultdict
from matplotlib import pyplot
from matplotlib import patches


REF_READ_ID = "REFERENCE"

# Cigar codes
REF = 0
SNP = 1
INS = 2
DEL = 3

# Indexes of reference_key list, the combination of keys serve as pointers to node objects in graph
POSITION = 0
CIGAR = 1
SEQUENCE = 2


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

        self.mapped_reads = dict()

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
        # self.positional_insert_lengths = dict()

        self.paths = defaultdict(list)  # path of keys that point to each node

        self.ploidy = ploidy
        self.graph = dict()

        # self.max_alleles_per_type = [0, 0, 0, 0]
        self.positional_n_alleles_per_type = dict()

        self.default_y_position_per_cigar = [0, -1, 2, 1]   # [REF, SNP, INS, DEL]

    def link_nodes(self, node1, node2):
        if node1 is not None and node2 is not None:
            node1.next_nodes.add(node2)
            node2.prev_nodes.add(node1)

    def delete_node(self, node, splice=False):
        p = 0
        n = 0

        for p,prev_node in enumerate(node.prev_nodes):
            prev_node.next_nodes.remove(node)

        for n,next_node in enumerate(node.next_nodes):
            next_node.prev_nodes.remove(node)

        if splice:
            if n > 0 and p > 0:
                for prev_node in node.prev_nodes:
                    for next_node in node.next_nodes:
                        prev_node.next_nodes.add(next_node)
                        next_node.prev_nodes.add(prev_node)

            for read_id, index in node.mapped_reads.items():
                print("DELETING:",node)
                self.remove_path_element(read_id=read_id, index=index)

        position = node.position
        cigar_code = node.cigar_code
        sequence = node.sequence

        # Erase node object
        self.graph[position][cigar_code][sequence] = None

        # Remove reference from dictionary
        del self.graph[position][cigar_code][sequence]

    def remove_path_element(self, read_id, index):
        # needs to handle end cases where index = 0 or index = len
        path = self.paths[read_id]

        print(path[index])
        print("before",len(path))
        self.paths[read_id] = path[:index] + path[index+1:]
        print("after",len(path[:index] + path[index+1:]))

    # def update_positional_insert_length(self, position, sequence):
    #     if position not in self.positional_insert_lengths:
    #         self.positional_insert_lengths[position] = len(sequence)
    #     elif len(sequence) > self.positional_insert_lengths[position]:
    #         self.positional_insert_lengths[position] = len(sequence)

    def initialize_position(self, position):
        template = [{}, {}, {}, {}]
        self.graph[position] = template

    def update_position(self, read_id, position, sequence, cigar_code, coverage=1):
        if position not in self.graph:
            self.initialize_position(position=position)

        if len(self.paths[read_id]) > 0:
            prev_node_keys = self.paths[read_id][-1]
            prev_position, prev_cigar, prev_sequence = prev_node_keys
            prev_node = self.graph[prev_position][prev_cigar][prev_sequence]
        else:
            prev_node = None

        if sequence not in self.graph[position][cigar_code]:
            # if cigar_code == INS:
                # Keep track of inserts for pileup printing purposes
                # self.update_positional_insert_length(position, sequence)

            node = Node(position=position,
                        cigar_code=cigar_code,
                        sequence=sequence,
                        coverage=coverage)

            self.graph[position][cigar_code][sequence] = node

        else:
            node = self.graph[position][cigar_code][sequence]
            node.coverage += 1

        # Make pointers between nodes
        if prev_node is not None:
            self.link_nodes(node1=prev_node, node2=node)

        # Get read path
        path = self.paths[read_id]

        # Add reference to node path to the node itself (god this is convoluted)
        self.graph[position][cigar_code][sequence].mapped_reads[read_id] = len(path)

        # Append reference to node to read path
        reference_keys = [position, cigar_code, sequence]
        path.append(reference_keys)

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

    # def get_insert_length(self, position):
    #     length = 0
    #     if position in self.positional_insert_lengths:
    #         length += self.positional_insert_lengths[position]
    #
    #     return length

    def get_insert_length(self, position):
        max_length = 0
        for sequence in self.graph[position][INS]:
            max_length = max(len(sequence), max_length)

        return max_length

    def get_insert_status_of_next_node(self, path, index):
        """
        Inserts are assigned to the positional graph based on their anchor position, so when printing the pileup,
        multiple nodes may be in the same position if there is an insert. To determine whether a sequence element should
        be followed by a filler character, it is necessary to know whether the next node is an insert, but the data
        contained in a node is not sufficient to learn this.
        :param path:
        :param index:
        :return:
        """
        status = False

        if index < len(path) -1:
            next_node_keys = path[index+1]
            next_position, next_cigar, next_sequence = next_node_keys
            next_node = self.graph[next_position][next_cigar][next_sequence]

            if next_node.cigar_code == INS:
                status = True

        return status

    def generate_pileup(self):
        # total_inserts = sum([l for l in self.positional_insert_lengths.values()])
        total_length = self.length + 20
        max_coverage = max([c for c in self.positional_coverage.values()])

        insert_lengths = list()
        for position in range(self.start_position, self.end_position+1):
            insert_lengths.append(str(self.get_insert_length(position)))

        pileup = [['_' for i in range(total_length+1)] for j in range(max_coverage+1)]

        for p,path in enumerate(self.paths.values()):
            index = path[0][POSITION] - self.start_position
            offset = 0

            for reference_key in path:
                position, cigar_code, sequence = reference_key

                try:
                    node = self.graph[position][cigar_code][sequence]
                except(KeyError):
                    continue

                # Skip this position if not in the range specified by user
                if node.position < self.start_position or node.position > self.end_position:
                    continue

                # position = node.position
                # sequence = node.sequence
                # cigar_code = node.cigar_code
                insert_length = self.get_insert_length(position)

                try:
                    next_is_insert = self.get_insert_status_of_next_node(path, index)
                except(KeyError):
                    next_is_insert = False

                # If theres an insert, then iterate through it as though each character was a node
                # for each character, increment a temporary offset +1, and increment global offset AFTER iterating
                #   all characters using positional_insert_lengths
                # If there's no insert, use positional_insert_lengths to increment the offset AFTER iterating the node

                # Insert at this position
                if cigar_code == INS:
                    for c,character in enumerate(sequence):
                        pileup[p][index+offset+c] = character

                    insert_difference = insert_length - len(sequence)
                    for i in range(0, insert_difference):
                        pileup[p][index+offset+c+i+1] = '*'

                    offset += insert_length - 1

                # Not an insert character, but could be in an anchor position that is followed by
                else:
                    pileup[p][index+offset] = sequence

                    if insert_length > 0 and not next_is_insert:
                        for i in range(insert_length):
                            pileup[p][index+offset+i+1] = '*'

                        offset += insert_length
                index += 1

        read_strings = ([''.join(row) for row in pileup])
        pileup_string = '\n'.join(read_strings)

        return pileup_string

    def clean_graph(self):
        figure, (axes1, axes2) = pyplot.subplots(nrows=2)

        self.plot_alignment_graph(axes=axes1, show=False)

        for position in range(self.start_position, self.end_position+1):
            for cigar_code in [REF, SNP, DEL, INS]:
                sequences = [key for key in self.graph[position][cigar_code]]

                for s,sequence in enumerate(sequences):
                    node = self.graph[position][cigar_code][sequence]
                    relative_coverage = node.coverage/self.positional_coverage[position]

                    if cigar_code == INS and relative_coverage < 0.2:
                        self.delete_node(self.graph[position][cigar_code][sequence])

                    # elif cigar_code != DEL and random.random() > tanh(relative_coverage):
                    #     self.delete_node(self.graph[position][cigar_code][sequence])

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
                    x_offset += max(1, max([len(seq) for seq in [node.sequence for node in nodes]])/2)

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

# Solutions to the problem of deleting and re-assigning nodes in a path:
#   Make a separate path object
#       Path objects can each have a mapping of keys to the index of the the corresponding node, which is updated when
#       nodes are deleted

#       ISSUE: what does the key refer to if the node is deleted? Erase key?
#       ISSUE: all remaining keys must also be updated to reflect the earlier change

#   Make paths a linked list of pointers to nodes, and have pointers to each pointer addressable by pos:cigar:seq
#


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
