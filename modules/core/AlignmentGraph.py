import random
import math
import numpy
import sys
from collections import defaultdict
from matplotlib import pyplot
from matplotlib import patches

CUTOFF = 0.1

REF_READ_ID = "REFERENCE"

NODE_FREQUENCY_THRESHOLD = 0.25

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
    def __init__(self, position, cigar_code, prev_node=None, next_node=None, sequence=None):
        # transition keys will be based on position:cigar_code:sequence which is always unique
        self.prev_nodes = set()
        self.next_nodes = set()

        self.position = position
        self.cigar_code = cigar_code
        self.sequence = sequence
        self.coverage = 0

        self.n = None
        self.true_variant = False

        # self.coordinate = None

        self.mapped_reads = dict()

    def __str__(self):
        return ' '.join(map(str, [self.position, self.cigar_code, self.sequence]))

    def __hash__(self):
        key = (self.position, self.cigar_code, self.sequence)
        return hash(key)


class AlignmentGraph:
    def __init__(self, chromosome_name, start_position, end_position, ploidy=2):
        # Predefined constants
        self.chromosome_name = chromosome_name
        self.start_position = start_position
        self.end_position = end_position
        self.length = self.end_position - self.start_position
        self.ploidy = ploidy

        # Positional data
        self.positional_reference = defaultdict(int)
        self.positional_alleles = defaultdict(int)
        self.positional_coverage = defaultdict(lambda: 0)
        self.positional_node_count = defaultdict(lambda: 0)

        self.paths = defaultdict(list)  # path of keys that point to each node
        self.path_index_offsets = defaultdict(list)
        self.graph = dict()

        # Plotting
        self.node_coordinates = dict()  # for plotting the graph
        self.positional_n_alleles_per_type = dict()
        self.default_y_position_per_cigar = [0, -1, 2, 1]   # [REF, SNP, INS, DEL]

        # Adjacency matrix
        self.n_nodes = 0

        self.cigar_code_names = ["REF", "SNP", "INS", "DEL"]

    def get_read_path_element_index(self, read_id, index):
        adjusted_index = int(index)

        if read_id in self.path_index_offsets:
            for item in self.path_index_offsets[read_id]:
                i, offset = item

                if i < index:
                    adjusted_index += offset

        return adjusted_index

    def print_path(self, read_id):
        path_string_list = list()
        for path_element in self.paths[read_id]:
            position_string = str(path_element[0])
            cigar_string = self.cigar_code_names[path_element[1]]
            sequence_string = path_element[2]

            path_string_list.append(position_string)
            path_string_list.append(cigar_string)
            path_string_list.append(sequence_string)
            path_string_list.append("->")

        print(' '.join(path_string_list))

    def reassign_node(self, node1, node2, delete=False):
        """
        Switch the identity of node1 to node2 so that all reads previously mapped to node1 it now map to node2,
        optionally delete node1
        :param node1:
        :param node2:
        :return:
        """
        node2_position = node2.position
        node2_cigar_code = node2.cigar_code
        node2_sequence = node2.sequence

        for path_element in node1.mapped_reads.items():
            # Reassign all paths that refer to this node
            # Reassign previous and next node linkages, and ensure that the destination node has been linked to node1's
            #   linkages
            # Optionally delete node1

            read_id, path_index = path_element

            adjusted_path_index = self.get_read_path_element_index(read_id=read_id, index=path_index)

            # print(read_id)
            # self.print_path(read_id)

            self.paths[read_id][adjusted_path_index] = [node2_position, node2_cigar_code, node2_sequence]

            # self.print_path(read_id)

            node2.mapped_reads[read_id] = path_index

            node2.coverage += 1
            node1.coverage -= 1

            # print(node1.coverage, node2.coverage)

        prev_nodes = node1.prev_nodes
        next_nodes = node1.next_nodes

        # if len(prev_nodes) > 0:
        for prev_node in prev_nodes:
            try:
                prev_node.next_nodes.remove(node1)
                self.link_nodes(node1=prev_node, node2=node2)
            except(KeyError):
                pass

        # if len(next_nodes) > 0:
        for next_node in next_nodes:
            try:
                next_node.prev_nodes.remove(node1)
                self.link_nodes(node1=node2, node2=next_node)
            except(KeyError):
                pass

        if delete:
            self.delete_node(node1, dereference=False, splice=False)

    def link_nodes(self, node1, node2):
        if node1 is not None and node2 is not None:
            node1.next_nodes.add(node2)
            node2.prev_nodes.add(node1)

    def delete_node(self, node, dereference=True, splice=True):
        p = 0
        n = 0

        if dereference:
            for p,prev_node in enumerate(node.prev_nodes):
                prev_node.next_nodes.remove(node)

            for n,next_node in enumerate(node.next_nodes):
                next_node.prev_nodes.remove(node)

        if splice:
            if n > 0 and p > 0:
                for prev_node in node.prev_nodes:
                    for next_node in node.next_nodes:
                        position = next_node.position
                        cigar_code = next_node.cigar_code
                        sequence = next_node.sequence

                        prev_node.next_nodes.add(self.graph[position][cigar_code][sequence])
                        next_node.prev_nodes.add(self.graph[position][cigar_code][sequence])

            for read_id, index in node.mapped_reads.items():
                # print("DELETING:",node)
                self.remove_path_element(read_id=read_id, index=index)
        # else:
        #     for read_id, index in node.mapped_reads.items():
        #         self.paths[read_id] = None

        position = node.position
        cigar_code = node.cigar_code
        sequence = node.sequence

        # Adjust coverage at this position
        if node.cigar_code == INS:
            self.positional_node_count[position] -= 1
        else:
            self.positional_coverage[position] -= 1

        # Erase node object (is this necessary?)
        self.graph[position][cigar_code][sequence] = None

        # Remove reference from dictionary
        del self.graph[position][cigar_code][sequence]

    def remove_path_element(self, read_id, index):
        # needs to handle end cases where index = 0 or index = len
        path = self.paths[read_id]

        adjusted_index = self.get_read_path_element_index(read_id=read_id, index=index)

        # print(path[adjusted_index])
        # print("before",len(path))
        self.paths[read_id] = path[:adjusted_index] + path[adjusted_index+1:]
        # print("after",len(path[:adjusted_index] + path[adjusted_index+1:]))

        self.path_index_offsets[read_id].append([index,-1])

    def initialize_position(self, position):
        template = [{}, {}, {}, {}]
        self.graph[position] = template

    def update_position(self, read_id, position, sequence, cigar_code, increment_coverage=True):
        if position not in self.graph:
            self.initialize_position(position=position)

        prev_node = self.get_previous_node(read_id)

        # if node doesn't already exist, create it
        if sequence not in self.graph[position][cigar_code]:
            node = Node(position=position,
                        cigar_code=cigar_code,
                        sequence=sequence)

            self.graph[position][cigar_code][sequence] = node

            # update node's n index for generating adjacency matrix
            node.n = self.n_nodes

            # update total number of nodes
            self.n_nodes += 1

        else:
            node = self.graph[position][cigar_code][sequence]

        # update node coverage
        if increment_coverage:
            node.coverage += 1

        # make pointers between nodes
        if prev_node is not None:
            self.link_nodes(node1=prev_node, node2=node)

        # append the read path
        self.update_read_path(read_id=read_id, position=position, cigar_code=cigar_code, sequence=sequence)

        # update coverage stats for this position
        self.update_coverage(position=position, cigar_code=cigar_code)

    def get_previous_node(self, read_id):
        if len(self.paths[read_id]) > 0:
            prev_node_keys = self.paths[read_id][-1]
            prev_position, prev_cigar, prev_sequence = prev_node_keys
            prev_node = self.graph[prev_position][prev_cigar][prev_sequence]
        else:
            prev_node = None

        return prev_node

    def update_read_path(self, read_id, position, cigar_code, sequence):
        # Get read path
        path = self.paths[read_id]

        # Add reference to the node position in the read path (isn't safe for deleting nodes!!)
        self.graph[position][cigar_code][sequence].mapped_reads[read_id] = len(path)

        # Append reference to node to read path
        reference_keys = [position, cigar_code, sequence]
        path.append(reference_keys)

    def update_coverage(self, position, cigar_code):
        if cigar_code != INS:
            self.positional_coverage[position] += 1

        self.positional_node_count[position] += 1

    def get_normalized_node_frequencies(self, position, include_inserts=False):
        """
        Calculate the relative proportion of each node at a position. If include_inserts, take into account the fact
        that inserts and refs both map to the same position, and ignore ref nodes that precede inserts? I guess?
        :param position:
        :param include_inserts: boolean flag
        :return:
        """
        # frequencies = list()
        # nodes = list()
        node_frequencies_below = list()
        node_frequencies_above = list()

        cigar_codes = [REF, SNP, DEL]

        if include_inserts:
            cigar_codes.append(INS)

        for cigar_code in cigar_codes:
            for sequence in self.graph[position][cigar_code]:
                node = self.graph[position][cigar_code][sequence]
                coverage = node.coverage

                if include_inserts and cigar_code == REF:
                    # the number of inserts is the total nodes minus the positional coverage
                    coverage -= (self.positional_node_count[position] - self.positional_coverage[position])

                frequency = coverage / self.positional_coverage[position]

                if frequency > CUTOFF:
                    node_frequencies_above.append((node, frequency))
                else:
                    node_frequencies_below.append((node, frequency))

        return node_frequencies_below, node_frequencies_above

    def reassign_by_coverage(self, position):
        node_frequencies_below, node_frequencies_above = \
            self.get_normalized_node_frequencies(position, include_inserts=True)

        nodes_above, frequencies_above = zip(*node_frequencies_above)

        for n, (node, frequency) in enumerate(node_frequencies_below):
            # inserts don't need reassignment, they are removed, leaving the ref node
            if node.cigar_code == INS:
                self.delete_node(node, splice=True)
            else:
                choice = numpy.random.multinomial(1, frequencies_above)
                choice = int(choice.nonzero()[0])
                reassignment_node = nodes_above[choice]

                # print(node, "->", reassignment_node)
                # don't allow reassignment to a delete...
                # print(frequency)
                # print("REASSIGNING")
                # print(node, "->", reassignment_node)

                if reassignment_node.cigar_code != DEL:
                    self.reassign_node(node1=node, node2=reassignment_node, delete=True)

                pass

    def clean_graph(self, plot=False):
        if plot:
            figure, (axes1, axes2) = pyplot.subplots(nrows=2)

            self.plot_alignment_graph(axes=axes1, show=False)

        for position in range(self.start_position, self.end_position+1):
            self.reassign_by_coverage(position=position)

        if plot:
            self.plot_alignment_graph(axes=axes2, show=False)
            pyplot.show()

    def get_node_stats(self):
        # cigar_codes = [REF, SNP, INS, DEL]

        cigar_codes = [[],[]]
        frequencies = [[[],[]] for code in [REF, SNP, INS, DEL]]

        for position in range(self.start_position, self.end_position+1):
            position_frequencies = list()

            if position not in self.graph:
                continue

            for cigar_code in [REF, SNP, INS, DEL]:
                for sequence in self.graph[position][cigar_code]:
                    node = self.graph[position][cigar_code][sequence]
                    coverage = node.coverage

                    if cigar_code == REF:
                        # the number of inserts is the total nodes minus the positional coverage
                        coverage -= (self.positional_node_count[position] - self.positional_coverage[position])

                    frequency = coverage / self.positional_coverage[position]
                    position_frequencies.append(frequency)

                    cigar_codes[int(node.true_variant)].append(cigar_code)
                    frequencies[cigar_code][int(node.true_variant)].append(frequency)

        return cigar_codes, frequencies

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

        if index < len(path) - 1:
            next_node_keys = path[index+1]
            next_position, next_cigar, next_sequence = next_node_keys
            next_node = self.graph[next_position][next_cigar][next_sequence]

            if next_node.cigar_code == INS:
                status = True

        return status

    def get_insert_length(self, position):
        max_length = 0
        for sequence in self.graph[position][INS]:
            if len(sequence) > max_length:
                max_length = len(sequence)

        return max_length

    def get_cumulative_insert_length(self, position):
        cumulative_length = 0
        for pos in range(self.start_position, position):
            cumulative_length += self.get_insert_length(pos)

        return cumulative_length

    def generate_pileup(self):
        total_length = self.length + 50

        insert_lengths = list()
        for position in range(self.start_position, self.end_position+1):
            insert_lengths.append(str(self.get_insert_length(position)))

        pileup = [['_' for i in range(total_length+1)] for j in range(len(self.paths))]

        for p,path in enumerate(self.paths.items()):
            read_id, path = path

            # print(self.graph[path[0][0]][path[0][1]][path[0][2]])
            start_index = path[0][POSITION] - self.start_position + self.get_cumulative_insert_length(path[0][POSITION])
            index = int(start_index)
            offset = 0

            for node_index, node_keys in enumerate(path):
                position, cigar_code, sequence = node_keys

                # print(read_id, path)
                node = self.graph[position][cigar_code][sequence]

                # Skip this position if not in the range specified by user
                if node.position < self.start_position or node.position > self.end_position:
                    continue

                insert_length = self.get_insert_length(position)

                next_is_insert = self.get_insert_status_of_next_node(path, node_index)

                # If there's an insert, then iterate through it as though each character was a node
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

                # Not an insert character, but could be in an anchor node that is followed by an insert node in
                # the same position
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

    def get_positional_coverage(self, position):
        positional_coverage = 0
        for cigar_code in [REF, SNP, DEL, INS]:
            for sequence in self.graph[position][cigar_code]:
                node = self.graph[position][cigar_code][sequence]
                positional_coverage += node.coverage

        return positional_coverage

    def print_alignment_graph(self):
        for position in range(self.start_position, self.end_position+1):
            print(position,
                  "REF", [node.sequence for node in self.graph[position][REF].values()],
                  "SNP", [node.sequence for node in self.graph[position][SNP].values()],
                  "DEL", [node.sequence for node in self.graph[position][DEL].values()],
                  "INS", [node.sequence for node in self.graph[position][INS].values()])
        print()

    def calculate_coordinates_for_plot(self, reset=True):
        if reset:
            self.node_coordinates = dict()

        x_offset = 0
        for position in range(self.start_position, self.end_position+1):
            for cigar_code in [REF, SNP, DEL, INS]:
                nodes = self.graph[position][cigar_code].values()

                if cigar_code == INS and len(nodes) > 0:
                    x_offset += max(1, max([len(seq) for seq in [node.sequence for node in nodes]])/4)

                for n,node in enumerate(nodes):
                    x = position - self.start_position + x_offset
                    y = self.get_y_position(cigar_code, position, n)

                    self.node_coordinates[node] = [x,y]

    def plot_alignment_graph(self, axes=None, show=True, set_axis_limits=True):
        self.calculate_coordinates_for_plot()

        default_color = [0, 0, 0]
        variant_color = [0/255, 153/255, 96/255]

        weights = ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy']

        all_positions = range(self.start_position, self.end_position)
        total_coverage = sum(self.positional_coverage[p] for p in all_positions) / (self.length)

        if axes is None:
            axes = pyplot.axes()

        min_y = sys.maxsize
        max_y = -sys.maxsize

        x_offset = 0
        cigar_codes = [REF, SNP, DEL, INS]
        for position in range(self.start_position, self.end_position+1):
            if position not in self.graph:
                continue

            for cigar_code in cigar_codes:
                nodes = self.graph[position][cigar_code].values()

                if cigar_code == INS and len(nodes) > 0:
                    x_offset += max(1, max([len(seq) for seq in [node.sequence for node in nodes]])/2)

                for n,node in enumerate(nodes):
                    # x = position - self.start_position + x_offset
                    # y = self.get_y_position(cigar_code, position, n)
                    # node.coordinate = [x,y]

                    node_color = default_color
                    if node.true_variant:
                        node_color = variant_color

                    x,y = self.node_coordinates[node]

                    # plot sequence
                    weight_index = min(int(round(node.coverage/total_coverage))*5,5)
                    weight = weights[weight_index]
                    axes.text(x, y, node.sequence, ha="center", va="center", zorder=2, weight=weight)

                    # plot node shape
                    width = (node.coverage/total_coverage)*5
                    p = patches.Circle([x,y], radius=0.33, zorder=1, facecolor="w", edgecolor=node_color, alpha=1, linewidth=width)
                    axes.add_patch(p)

                    # plot edge that connects node with previous nodes
                    if len(node.prev_nodes) > 0:

                        for prev_node in node.prev_nodes:
                            x_prev, y_prev = self.node_coordinates[prev_node]
                            transition_weight = min([prev_node.coverage, node.coverage])
                            width = min(6*transition_weight/total_coverage, 6)

                            # print()
                            # print(prev_node, "->", node)

                            edge_color = default_color
                            if prev_node.true_variant and node.true_variant:
                                edge_color = variant_color

                            if y_prev != y:
                                p = patches.FancyArrowPatch(posA=[x_prev,y_prev], posB=[x,y], color=edge_color, zorder=0, lw=width, connectionstyle=patches.ConnectionStyle.Arc3(rad=-0.2))
                                axes.add_patch(p)
                            else:
                                axes.plot([x_prev,x], [y_prev,y], lw=width, color=edge_color, zorder=0, alpha=1)

                    if y < min_y:
                        min_y = y
                    if y > max_y:
                        max_y = y

        x_limits = [-1, self.length + x_offset + 1]
        y_limits = [min_y-1, max_y+1]

        if set_axis_limits:
            axes.set_xlim(-1, self.length + x_offset + 1)
            axes.set_ylim(min_y-1, max_y+1)
            axes.set_aspect("equal")

        if show:
            pyplot.show()

        return axes, x_limits, y_limits

    def generate_stepwise_incidence_matrix(self):
        pass

    def generate_adjacency_matrix(self, label_variants=False):
        cigar_codes = [REF, SNP, DEL, INS]

        # n_channels = 2

        matrix = numpy.zeros([self.n_nodes, self.n_nodes], dtype=numpy.uint8)

        for position in range(self.start_position, self.end_position+1):
            for cigar_code in cigar_codes:
                sequences = self.graph[position][cigar_code].keys()

                for sequence in sequences:
                    node = self.graph[position][cigar_code][sequence]
                    n1 = node.n

                    for next_node in node.next_nodes:
                        transition_weight = min(node.coverage, next_node.coverage)

                        if label_variants:
                            if node.true_variant and next_node.true_variant:
                                transition_weight = 1
                            else:
                                continue

                        n2 = next_node.n

                        if matrix[n1,n2] != 0:
                            print("WARNING: adjacency matrix overwrite conflict detected:", n1, n2)

                        matrix[n1,n2] = transition_weight
                        matrix[n2,n1] = transition_weight

        return matrix

        print()
        print(matrix)



# BUGS TO FIX!!
#   Switch to explicit edge objects?

# Solutions to the problem of deleting and re-assigning nodes in a path:
#   Make a separate path object
#       Path objects can each have a mapping of keys to the index of the the corresponding node, which is updated when
#       nodes are deleted

#       ISSUE: what does the key refer to if the node is deleted? Erase key?
#       ISSUE: all remaining keys must also be updated to reflect the earlier change

#   Make paths a linked list of pointers to nodes, and have pointers to each pointer addressable by pos:cigar:seq

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
