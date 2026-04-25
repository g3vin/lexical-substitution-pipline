from dataclasses import dataclass


TARGET_START = "<target>"
TARGET_END = "</target>"


@dataclass
class SubstituteCandidate:
    word: str
    pos_match: bool = False
    lexical_score: float = 0.0
    target_score: float = 0.0
    semantic_score: float = 0.0
    wic_score: float = 0.0
    mlm_score: float = 0.0
    substitute_score: float = 0.0
    rerank_score: float = 0.0
    stage1_score: float = 0.0
    final_score: float = 0.0


@dataclass
class RankerConfig:
    lexical_weight: float = 0.25
    target_weight: float = 0.15
    semantic_weight: float = 0.15
    mlm_weight: float = 0.25
    substitute_weight: float = 0.0
    rerank_weight: float = 0.20
    semantic_threshold: float = 0.75
    target_threshold: float = 0.15
    mlm_threshold: float = 0.16
    lexical_precision_bias: float = 1.0
    semantic_below_threshold_penalty: float = 0.55
    target_below_threshold_penalty: float = 0.85
    mlm_below_threshold_penalty: float = 0.85
    substitute_below_min_penalty: float = 0.45
    substitute_min_score: float = 0.15
    weak_source_substitute_min_score: float = 0.20
    substitute_low_score_threshold: float = 0.25
    substitute_low_score_penalty: float = 0.50
    wordnet_related_min_substitute_score: float = 0.45
    wordnet_related_penalty: float = 0.65
    rerank_pool_size: int = 24
    rerank_blend_alpha: float = 0.75
