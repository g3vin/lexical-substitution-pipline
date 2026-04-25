from typing import List

from .types import SubstituteCandidate


class RankingMixin:
    def _normalize_scores(self, values: List[float]) -> List[float]:
        if not values:
            return []

        minimum = min(values)
        maximum = max(values)

        if maximum - minimum < 1e-9:
            return [0.5 for _ in values]

        return [(value - minimum) / (maximum - minimum) for value in values]

    def _base_score_from_features(self, candidate: SubstituteCandidate) -> float:
        learned_ranker = getattr(self, "learned_ranker", None)

        if learned_ranker is not None:
            return float(
                learned_ranker.predict_score(
                    {
                        "lexical": candidate.lexical_score,
                        "target": candidate.target_score,
                        "semantic": candidate.semantic_score,
                        "mlm": candidate.mlm_score,
                        "substitute": candidate.substitute_score,
                        "rerank": candidate.rerank_score,
                    }
                )
            )

        score = (
            self.ranker_config.lexical_weight * candidate.lexical_score
            + self.ranker_config.target_weight * candidate.target_score
            + self.ranker_config.semantic_weight * candidate.semantic_score
            + self.ranker_config.mlm_weight * candidate.mlm_score
            + self.ranker_config.substitute_weight * candidate.substitute_score
        )

        if candidate.lexical_score < 1.0:
            score *= self.ranker_config.lexical_precision_bias

        if candidate.semantic_score < self.ranker_config.semantic_threshold:
            score *= self.ranker_config.semantic_below_threshold_penalty

        if candidate.target_score < self.ranker_config.target_threshold:
            score *= self.ranker_config.target_below_threshold_penalty

        if candidate.mlm_score < self.ranker_config.mlm_threshold:
            score *= self.ranker_config.mlm_below_threshold_penalty

        if self._has_substitute_model() and (
            candidate.substitute_score < self._candidate_substitute_min_score(candidate)
        ):
            score *= self.ranker_config.substitute_below_min_penalty

        if (
            self._has_substitute_model()
            and candidate.substitute_score
            < self.ranker_config.substitute_low_score_threshold
        ):
            score *= self.ranker_config.substitute_low_score_penalty

        if self._should_penalize_wordnet_related(candidate):
            score *= self.ranker_config.wordnet_related_penalty

        return score

    def _score_stage_one(
        self,
        candidates: List[SubstituteCandidate],
    ) -> List[SubstituteCandidate]:
        for candidate in candidates:
            self.promote_high_confidence_synonym(candidate)
            candidate.stage1_score = self._base_score_from_features(candidate)
            candidate.final_score = candidate.stage1_score

        return sorted(candidates, key=lambda x: x.stage1_score, reverse=True)

    # ---------------------------------------------------------
    # 5. Cross-encoder reranking
    # ---------------------------------------------------------

    def rerank(
        self,
        sentence: str,
        target_word: str,
        candidates: List[SubstituteCandidate],
        target_offset: int | None = None,
        pool_size: int | None = None,
    ) -> List[SubstituteCandidate]:
        if not candidates:
            return candidates

        for candidate in candidates:
            candidate.rerank_score = 0.0

        if self.cross_encoder is None:
            return candidates

        ranked_candidates = sorted(
            candidates,
            key=lambda candidate: candidate.stage1_score,
            reverse=True,
        )
        if pool_size is None:
            rerank_candidates = ranked_candidates
        else:
            rerank_candidates = ranked_candidates[: max(0, pool_size)]

        if not rerank_candidates:
            return candidates

        pairs = []

        for candidate in rerank_candidates:
            substituted = self.replace_target(
                sentence,
                target_word,
                candidate.word,
                target_offset,
            )
            pairs.append((sentence, substituted))

        scores = self.cross_encoder.predict(pairs)

        for candidate, score in zip(rerank_candidates, scores):
            candidate.rerank_score = float(score)

        return candidates


    # final weighted score

    def compute_final_scores(
        self,
        candidates: List[SubstituteCandidate],
        rerank_pool_size: int | None = None,
        rerank_blend_alpha: float | None = None,
    ) -> List[SubstituteCandidate]:
        if not candidates:
            return candidates

        rerank_pool_size = (
            self.ranker_config.rerank_pool_size
            if rerank_pool_size is None
            else rerank_pool_size
        )
        rerank_blend_alpha = (
            self.ranker_config.rerank_blend_alpha
            if rerank_blend_alpha is None
            else rerank_blend_alpha
        )

        ranked_candidates = self._score_stage_one(candidates)
        stage1_scores = [candidate.stage1_score for candidate in ranked_candidates]
        stage1_norm = self._normalize_scores(stage1_scores)

        for candidate, normalized_score in zip(ranked_candidates, stage1_norm):
            candidate.final_score = normalized_score

        if self.cross_encoder is None or rerank_pool_size <= 0:
            return ranked_candidates

        reranked_candidates = ranked_candidates[: max(0, rerank_pool_size)]

        rerank_norm = self._normalize_scores(
            [candidate.rerank_score for candidate in reranked_candidates]
        )
        stage1_pool_norm = self._normalize_scores(
            [candidate.stage1_score for candidate in reranked_candidates]
        )

        for candidate, base_norm, rerank_score in zip(
            reranked_candidates,
            stage1_pool_norm,
            rerank_norm,
        ):
            candidate.final_score = 1.0 + (
                (1.0 - rerank_blend_alpha) * base_norm
                + rerank_blend_alpha * rerank_score
            )

        return sorted(candidates, key=lambda x: x.final_score, reverse=True)

    def _should_penalize_wordnet_related(self, candidate: SubstituteCandidate) -> bool:
        sources = self.candidate_sources.get(candidate.word.lower(), set())

        if "wordnet_related" not in sources:
            return False

        if "swords" in sources or "wordnet" in sources:
            return False

        return (
            candidate.substitute_score
            < self.ranker_config.wordnet_related_min_substitute_score
        )

    def promote_high_confidence_synonym(self, candidate: SubstituteCandidate):
        if candidate.lexical_score >= 1.0:
            return

        if (
            candidate.target_score >= 0.70
            and candidate.semantic_score >= 0.90
            and candidate.substitute_score >= 0.60
        ):
            candidate.lexical_score = max(candidate.lexical_score, 0.90)
