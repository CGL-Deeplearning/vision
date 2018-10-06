import re

class Read:
    def __init__(self, read):
        self.query_name = read.query_name
        self.mapping_quality = read.mapping_quality
        self.is_secondary = read.is_secondary
        self.is_supplementary = read.is_supplementary
        self.is_unmapped = read.is_unmapped
        self.is_qcfail = read.is_qcfail
        self.query_qualities = read.query_qualities
        self.is_reverse = read.is_reverse
        self.reference_positions = read.get_reference_positions()

        self.query_sequence = read.query_sequence
        self.reference_start = read.reference_start
        self.reference_end = read.reference_end
        self.cigartuples = read.cigartuples
        self.cigar = read.cigarstring

    def cigar_op_to_int(self, cigar_op):
        if cigar_op == 'M':
            return 0
        if cigar_op == 'I':
            return 1
        if cigar_op == 'D':
            return 2
        if cigar_op == 'N':
            return 3
        if cigar_op == 'S':
            return 4
        if cigar_op == 'H':
            return 5
        if cigar_op == 'P':
            return 6
        if cigar_op == '=':
            return 7
        if cigar_op == 'X':
            return 8
        if cigar_op == 'B':
            return 9

    def parse_cigar(self, cigar):
        literal_cigar_tuples = re.findall(r'(\d+)([A-Z]{1})', cigar)
        cigar_tuples = []
        for cigar_len, cigar_op in literal_cigar_tuples:
            cigar_tuples.append((self.cigar_op_to_int(cigar_op), int(cigar_len)))

        return cigar_tuples

    def refine_with_alignment(self, alignment, ref_start):
        self.query_sequence = alignment.query
        self.reference_start = ref_start+alignment.reference_begin
        self.reference_end = ref_start + alignment.reference_end
        self.cigartuples = self.parse_cigar(alignment.cigar)
