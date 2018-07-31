from modules.core.VCFGraphGenerator import VCFGraphGenerator
from modules.core.AlignmentGraph import AlignmentGraph


class KmerGraphLabeler:
    def __init__(self, kmer_graph, haplotypes):
        self.haplotypes = haplotypes
        self.kmer_graph = kmer_graph
        self.k = kmer_graph.k

    def parse_region(self):
        print(self.kmer_graph.graph.keys())

        for haplotype in self.haplotypes:
            length = len(haplotype)
            n = length - self.k + 1

            for i in range(n):
                kmer = haplotype[i:i+self.k]

                if kmer in self.kmer_graph.graph:
                    self.kmer_graph.graph[kmer].true_variant = True


