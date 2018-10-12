from collections import defaultdict
from graphviz import Digraph
import copy
from modules.core.OptionValues import MIN_BASE_QUALITY_FOR_CANDIDATE, MIN_EDGE_SUPPORT, MAX_ALLOWED_PATHS
"""doing this: https://software.broadinstitute.org/gatk/documentation/article?id=11076"""


class DeBruijnGraph:
    def __init__(self, ref, reads, k, start_pos, end_pos):
        self.ref = ref
        self.reads = reads
        self.kmer_size = k
        self.hash = defaultdict(int)
        self.hash_value = 1
        self.opposite_hash = defaultdict(int)
        self.in_nodes = defaultdict(list)
        self.out_nodes = defaultdict(list)
        self.edge_weights = defaultdict(defaultdict)

        self.start_pos = start_pos
        self.end_pos = end_pos

        # unpruned nodes
        self.legit_nodes = list()

    def get_vertex_hash_from_kmer(self, kmer):
        if kmer in self.hash:
            return self.hash[kmer]

        self.hash[kmer] = self.hash_value
        self.opposite_hash[self.hash_value] = kmer
        self.hash_value += 1

        return self.hash[kmer]

    def visualize(self, output_filename, on_pruned_nodes=False):
        dot = Digraph(comment='dbruijn graph')
        for i in range(1, self.hash_value):
            if on_pruned_nodes is True and i not in self.legit_nodes:
                continue
            kmer = self.opposite_hash[i]
            dot.node(str(i), label=kmer)

        for i in range(1, self.hash_value):
            if on_pruned_nodes is True and i not in self.legit_nodes:
                continue
            for j in range(len(self.out_nodes[i])):
                node_a = i
                node_b = self.out_nodes[i][j]
                if on_pruned_nodes is True and node_b not in self.legit_nodes:
                    continue
                weight = self.edge_weights[node_a][node_b]
                dot.edge(str(node_a), str(node_b), label=str(weight))

        # print(self.dot.source)
        dot.render('outputs/'+output_filename+'.sv')

    def prune_nodes(self, source_, sink_):
        stack_node = list()
        stack_node.append(source_)
        forward_visit_nodes = list()

        # we are sure that this graph is acyclic, no cyclic graph will pass here
        while True:
            if not stack_node:
                break
            current_node = stack_node.pop()
            forward_visit_nodes.append(current_node)
            for j in range(len(self.out_nodes[current_node])):
                new_node = self.out_nodes[current_node][j]
                if new_node not in forward_visit_nodes and current_node != sink_:
                    stack_node.append(new_node)

        backward_visit_nodes = list()
        backward_stack_node = list()
        backward_stack_node.append(sink_)

        # we are sure that this graph is acyclic, no cyclic graph will pass here
        while True:
            if not backward_stack_node:
                break
            current_node = backward_stack_node.pop()
            backward_visit_nodes.append(current_node)

            for j in range(len(self.in_nodes[current_node])):
                new_node = self.in_nodes[current_node][j]
                if new_node not in backward_visit_nodes and current_node != source_:
                    backward_stack_node.append(new_node)

        legit_nodes = list(set(forward_visit_nodes) & set(backward_visit_nodes))
        return legit_nodes

    def prune_graph(self, source_, sink_):
        unpruned_weights = copy.deepcopy(self.edge_weights)
        # prune edges
        for node_a in unpruned_weights:
            for node_b in unpruned_weights[node_a]:
                if unpruned_weights[node_a][node_b][1] is False and \
                        unpruned_weights[node_a][node_b][0] < MIN_EDGE_SUPPORT:

                    self.in_nodes[node_b].remove(node_a)
                    self.out_nodes[node_a].remove(node_b)
                    del self.edge_weights[node_a][node_b]

        # do a forward dfs and color the nodes visited
        self.legit_nodes = self.prune_nodes(source_, sink_)

    def get_haplotypes(self, source_, sink_):
        finished_paths = list()
        running_paths = list()
        running_paths.append([source_])

        while running_paths:
            if len(finished_paths) + len(running_paths) > MAX_ALLOWED_PATHS:
                return []
            # get the front path
            current_path = running_paths.pop(0)
            running_node = current_path[-1]
            for i in range(len(self.out_nodes[running_node])):
                next_node = self.out_nodes[running_node][i]
                if next_node not in self.legit_nodes:
                    continue
                if next_node == sink_:
                    finished_paths.append(current_path + [next_node])
                else:
                    running_paths.append(current_path + [next_node])

        # get haplotypes from paths
        haplotype_set = []
        for path in finished_paths:
            haplotype_str = self.opposite_hash[path[0]]
            for i in range(1, len(path)):
                haplotype_str += self.opposite_hash[path[i]][-1]
            haplotype_set.append(haplotype_str)

        return haplotype_set

    def dfs_cycle_finder(self, source_, sink_):
        stack_node = list()
        stack_node.append(source_)
        visit_color = defaultdict(bool)

        while sink_ not in visit_color:
            if not stack_node:
                break
            current_node = stack_node.pop()
            visit_color[current_node] = True
            for j in range(len(self.out_nodes[current_node])):
                new_node = self.out_nodes[current_node][j]
                if new_node in visit_color:
                    # back node, has a cycle
                    return True
                stack_node.append(new_node)

        return False

    def add_read_seq_to_graph(self, read_seq):
        prev_vertex_hash = self.get_vertex_hash_from_kmer(read_seq[0:self.kmer_size])

        for i in range(1, len(read_seq) - self.kmer_size + 1):
            kmer = read_seq[i:i + self.kmer_size]
            current_vertex_hash = self.get_vertex_hash_from_kmer(kmer)

            if prev_vertex_hash in self.edge_weights and current_vertex_hash in self.edge_weights[prev_vertex_hash]:
                self.edge_weights[prev_vertex_hash][current_vertex_hash][0] += 1
            else:
                self.edge_weights[prev_vertex_hash][current_vertex_hash] = [1, False]
                self.in_nodes[current_vertex_hash].append(prev_vertex_hash)
                self.out_nodes[prev_vertex_hash].append(current_vertex_hash)

            prev_vertex_hash = current_vertex_hash

    @staticmethod
    def check_base_ok(base):
        if base != 'A' and base != 'C' and base != 'G' and base != 'T':
            return False
        return True

    def create_graph(self):
        prev_vertex_hash = self.get_vertex_hash_from_kmer(self.ref[0:self.kmer_size])
        source_ = prev_vertex_hash
        for i in range(1, len(self.ref) - self.kmer_size + 1):
            kmer = self.ref[i:i+self.kmer_size]
            current_vertex_hash = self.get_vertex_hash_from_kmer(kmer)

            self.in_nodes[current_vertex_hash].append(prev_vertex_hash)
            self.out_nodes[prev_vertex_hash].append(current_vertex_hash)
            self.edge_weights[prev_vertex_hash][current_vertex_hash] = [0, True]

            prev_vertex_hash = current_vertex_hash
        sink_ = prev_vertex_hash
        # self.visualize('Reference_graph', False)
        for read in self.reads:
            read_start = read.reference_start
            read_seq = read.query_sequence
            base_quals = read.query_qualities
            read_start_index = 0
            if read_start < self.start_pos:
                read_start_index = self.start_pos - read_start
            read_seq = read_seq[read_start_index:]
            base_quals = base_quals[read_start_index:]

            if len(read_seq) < self.kmer_size + 1:
                continue

            next_bad_qual = [index for index, base_qual in enumerate(base_quals)
                             if base_qual < MIN_BASE_QUALITY_FOR_CANDIDATE]
            next_bad_base = [index for index, read_base in enumerate(read_seq) if not self.check_base_ok(read_base)]
            bad_base_indices = sorted(next_bad_qual + next_bad_base + [len(read_seq) + 1])

            last_base = len(read_seq) - self.kmer_size
            current_position = 0
            bad_position_index = 0
            while current_position < last_base:
                bad_position = bad_base_indices[bad_position_index]
                read_sub_seq = read_seq[current_position:bad_position-1]
                if len(read_sub_seq) > self.kmer_size:
                    self.add_read_seq_to_graph(read_sub_seq)
                current_position = bad_position + 1
                bad_position_index += 1

        # self.visualize_graph()
        return self.dfs_cycle_finder(source_, sink_), source_, sink_


class DeBruijnGraphCreator:
    def __init__(self, start_pos, end_pos):
        self.region_start_position = start_pos
        self.region_end_position = end_pos

    @staticmethod
    def find_min_k_from_ref(ref, bounds):
        bound_min_k, bound_max_k, bound_step_k = bounds
        min_k = None
        max_k = min(bound_max_k, len(ref) - 1)

        for k in range(bound_min_k, bound_max_k + 1, bound_step_k):
            has_cycle = False
            kmers = set()
            for i in range(0, len(ref) - k + 1):
                kmer = ref[i:i+k]
                if kmer in kmers:
                    has_cycle = True
                    break
                else:
                    kmers.add(kmer)

            if not has_cycle:
                min_k = k
                break

        return min_k, max_k

    def find_haplotypes_through_linear_search_over_kmer(self, ref, reads, bounds):
        bound_min_k, bound_max_k, bound_step_k = bounds
        for k in range(bound_min_k, bound_max_k + 1, bound_step_k):
            graph = DeBruijnGraph(ref, reads, k, self.region_start_position, self.region_end_position)
            has_cycle, source_, sink_ = graph.create_graph()
            if has_cycle:
                continue
            else:
                # graph.visualize('Before_pruning', False)
                graph.prune_graph(source_, sink_)
                return graph.get_haplotypes(source_, sink_)
                # graph.visualize('Final_pruned', True)

        return []
