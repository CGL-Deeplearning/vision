# DEBRUIJN GRAPH OPTIONS
class DeBruijnGraphOptions(object):
    MIN_K = 10
    MAX_K = 100
    STEP_K = 1
    MIN_EDGE_SUPPORT = 2
    MAX_ALLOWED_PATHS = 256

    # base and map quality
    MIN_BASE_QUALITY = 20
    MIN_MAP_QUALITY = 20


class ActiveRegionOptions(object):
    MIN_MAPQ = 20
    MIN_BASE_QUALITY = 20
    MIN_REGION_SIZE = 80
    MAX_ACTIVE_REGION_SIZE = 1000

    # LINEAR MODEL WEIGHTS FOR ACTIVE REGION FINDING
    MATCH_WEIGHT = -0.06
    MISMATCH_WEIGHT = -0.09
    INSERT_WEIGHT = 2.5
    DELETE_WEIGHT = 1.8
    SOFT_CLIP_WEIGHT = 3.0
    THRESHOLD_VALUE = 3.5


# ALIGNER OPTIONS
class AlingerOptions(object):
    SEED_K_MER_SIZE = 23
    MATCH_PENALTY = 4
    MISMATCH_PENALTY = 6
    GAP_OPEN_PENALTY = 8
    GAP_EXTEND_PENALTY = 2
    ALIGNMENT_SAFE_BASES = 20


# CANDIDATE FINDER OPTIONS
class CandidateFinderOptions(object):
    MIN_BASE_QUALITY = 15
    MIN_MAP_QUALITY = 15
    MIN_MISMATCH_THRESHOLD = 1
    MIN_MISMATCH_PERCENT_THRESHOLD_SNP = 12
    MIN_MISMATCH_PERCENT_THRESHOLD_INDEL = 12

