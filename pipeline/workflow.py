from collections import defaultdict
from typing import List

from .types import RankerConfig, SubstituteCandidate


class WorkflowMixin:
    # full pipeline

    def substitute(
        self,
        sentence: str,
        target_word: str,
        top_k: int = 20,
        target_offset: int | None = None,
    ) -> List[SubstituteCandidate]:
        raw_candidates = self.generate_candidates(sentence, target_word)

        candidates = self.linguistic_filter(
            sentence,
            target_word,
            raw_candidates,
            target_offset=target_offset,
        )

        candidates = self.semantic_filter(
            sentence,
            target_word,
            candidates,
            target_offset=target_offset,
        )

        candidates = self.substitute_validity_filter(
            sentence,
            target_word,
            candidates,
            threshold=self.substitute_threshold if self.substitute_hard_filter else 0.0,
            target_offset=target_offset,
        )

        candidates = self.contradiction_filter(
            sentence,
            target_word,
            candidates,
            target_offset=target_offset,
        )

        candidates = self.mlm_score(
            sentence,
            target_word,
            candidates,
            target_offset=target_offset,
        )

        candidates = self.plausibility_filter(candidates)
        candidates = self.compute_final_scores(candidates, rerank_pool_size=0)

        candidates = self.rerank(
            sentence,
            target_word,
            candidates,
            target_offset=target_offset,
            pool_size=self.ranker_config.rerank_pool_size,
        )

        candidates = self.compute_final_scores(candidates)

        results = candidates[:top_k]

        if self.preserve_morphology:
            results = self.apply_morphology(sentence, target_word, results, target_offset)

        return results

    def score_substitutes(
        self,
        sentence: str,
        target_word: str,
        substitutes: List[str],
        target_offset: int | None = None,
        apply_thresholds: bool = True,
    ) -> List[SubstituteCandidate]:
        self.candidate_sources = defaultdict(set)

        for substitute in substitutes:
            self.candidate_sources[substitute.lower()].add("given")

        original_config = self.ranker_config
        if not apply_thresholds:
            self.ranker_config = RankerConfig(
                **{
                    **self.ranker_config.__dict__,
                    "semantic_threshold": 0.0,
                    "target_threshold": 0.0,
                    "mlm_threshold": 0.0,
                }
            )

        candidates = self.linguistic_filter(
            sentence,
            target_word,
            substitutes,
            target_offset=target_offset,
        )
        candidates = self.semantic_filter(
            sentence,
            target_word,
            candidates,
            target_offset=target_offset,
        )
        candidates = self.substitute_validity_filter(
            sentence,
            target_word,
            candidates,
            threshold=(
                self.substitute_threshold
                if apply_thresholds and self.substitute_hard_filter
                else 0.0
            ),
            target_offset=target_offset,
        )
        candidates = self.contradiction_filter(
            sentence,
            target_word,
            candidates,
            target_offset=target_offset,
        )
        candidates = self.mlm_score(
            sentence,
            target_word,
            candidates,
            target_offset=target_offset,
        )
        if apply_thresholds:
            candidates = self.plausibility_filter(candidates)

        candidates = self.compute_final_scores(candidates, rerank_pool_size=0)
        candidates = self.rerank(
            sentence,
            target_word,
            candidates,
            target_offset=target_offset,
            pool_size=None if not apply_thresholds else self.ranker_config.rerank_pool_size,
        )
        candidates = self.compute_final_scores(candidates)

        if not apply_thresholds:
            self.ranker_config = original_config

        return candidates
