from typing import List

import torch

from .types import SubstituteCandidate


class SemanticAndClassifierFilterMixin:
    # semantic filtering with Sentence-BERT

    def semantic_filter(
        self,
        sentence: str,
        target_word: str,
        candidates: List[SubstituteCandidate],
        threshold: float | None = None,
        target_offset: int | None = None,
    ) -> List[SubstituteCandidate]:
        threshold = self.ranker_config.semantic_threshold if threshold is None else threshold
        original_embedding = self.sbert.encode(sentence, convert_to_tensor=True)

        kept = []

        for candidate in candidates:
            substituted = self.replace_target(
                sentence,
                target_word,
                candidate.word,
                target_offset,
            )
            substitute_embedding = self.sbert.encode(substituted, convert_to_tensor=True)

            score = torch.nn.functional.cosine_similarity(
                original_embedding,
                substitute_embedding,
                dim=0,
            ).item()

            candidate.semantic_score = score

            if score >= threshold:
                kept.append(candidate)

        return kept


    # sense consistency filtering with a fine-tuned WiC classifier

    def wic_filter(
        self,
        sentence: str,
        target_word: str,
        candidates: List[SubstituteCandidate],
        threshold: float = 0.5,
        target_offset: int | None = None,
    ) -> List[SubstituteCandidate]:
        if self.wic_model is None or self.wic_tokenizer is None:
            return candidates

        if not candidates:
            return candidates

        original_marked = self.mark_target(sentence, target_word, target_offset)
        originals = []
        substitutions = []

        for candidate in candidates:
            substituted = self.replace_target(
                sentence,
                target_word,
                candidate.word,
                target_offset,
            )
            substitutions.append(self.mark_target(substituted, candidate.word, target_offset))
            originals.append(original_marked)

        inputs = self.wic_tokenizer(
            originals,
            substitutions,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.wic_model(**inputs).logits

        scores = self._wic_positive_scores(logits)
        kept = []

        for candidate, score in zip(candidates, scores):
            candidate.wic_score = score

            if score >= threshold:
                kept.append(candidate)

        return kept

    def _wic_positive_scores(self, logits: torch.Tensor) -> List[float]:
        if logits.shape[-1] == 1:
            return torch.sigmoid(logits[:, 0]).detach().cpu().tolist()

        positive_index = self._wic_positive_label_index(logits.shape[-1])
        probs = torch.softmax(logits, dim=-1)
        return probs[:, positive_index].detach().cpu().tolist()

    def _wic_positive_label_index(self, num_labels: int) -> int:
        if isinstance(self.wic_positive_label, int):
            return self.wic_positive_label

        if isinstance(self.wic_positive_label, str):
            label = self.wic_positive_label.lower()

            if label.isdigit():
                return int(label)

            id2label = getattr(self.wic_model.config, "id2label", {})
            for index, name in id2label.items():
                if str(name).lower() == label:
                    return int(index)

        id2label = getattr(self.wic_model.config, "id2label", {})
        positive_label_names = ("true", "same", "positive", "yes", "label_1", "1")

        for index, name in id2label.items():
            if str(name).lower() in positive_label_names:
                return int(index)

        return 1 if num_labels > 1 else 0


    # optional supervised substitute-validity filtering

    def substitute_validity_filter(
        self,
        sentence: str,
        target_word: str,
        candidates: List[SubstituteCandidate],
        threshold: float = 0.5,
        target_offset: int | None = None,
    ) -> List[SubstituteCandidate]:
        if self.substitute_model is None or self.substitute_tokenizer is None:
            return candidates

        if not candidates:
            return candidates

        originals = []
        substitutions = []
        original_marked = self.mark_target(sentence, target_word, target_offset)

        for candidate in candidates:
            substituted = self.replace_target(
                sentence,
                target_word,
                candidate.word,
                target_offset,
            )
            originals.append(original_marked)
            substitutions.append(self.mark_target(substituted, candidate.word, target_offset))

        inputs = self.substitute_tokenizer(
            originals,
            substitutions,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.substitute_model(**inputs).logits

        scores = self._classifier_positive_scores(
            logits,
            self.substitute_model,
            self.substitute_positive_label,
        )
        kept = []

        for candidate, score in zip(candidates, scores):
            candidate.substitute_score = score

            if score >= threshold:
                kept.append(candidate)

        return kept

    def contradiction_filter(
        self,
        sentence: str,
        target_word: str,
        candidates: List[SubstituteCandidate],
        target_offset: int | None = None,
        threshold: float | None = None,
    ) -> List[SubstituteCandidate]:
        if not candidates:
            return candidates

        doc = self.nlp(sentence)
        target_token = self._find_target_token(doc, target_word, target_offset)

        if target_token is None:
            return candidates

        kept = []

        for candidate in candidates:
            if self._is_candidate_opposite(
                sentence,
                target_word,
                target_token,
                candidate,
                target_offset,
            ):
                continue

            kept.append(candidate)

        if self.nli_model is None or self.nli_tokenizer is None:
            return kept

        threshold = (
            self.nli_contradiction_threshold if threshold is None else threshold
        )
        pairs = []

        for candidate in kept:
            surface = self._candidate_surface_in_context(candidate.word, target_token)
            substituted = self.replace_target(
                sentence,
                target_word,
                surface,
                target_offset,
            )
            pairs.append((sentence, substituted))
            pairs.append((substituted, sentence))

        inputs = self.nli_tokenizer(
            [premise for premise, _ in pairs],
            [hypothesis for _, hypothesis in pairs],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.nli_model(**inputs).logits

        contradiction_scores = self._nli_contradiction_scores(logits)
        filtered = []

        for index, candidate in enumerate(kept):
            forward = contradiction_scores[2 * index]
            backward = contradiction_scores[2 * index + 1]

            if max(forward, backward) < threshold:
                filtered.append(candidate)

        return filtered

    def _is_candidate_opposite(
        self,
        sentence: str,
        target_word: str,
        target_token,
        candidate: SubstituteCandidate,
        target_offset: int | None = None,
    ) -> bool:
        candidate_lemma = self._candidate_lemma_in_context(
            sentence,
            target_word,
            target_token,
            candidate.word,
            target_offset,
        )

        if candidate_lemma is None:
            return False

        return self._are_wordnet_opposites(
            target_token.lemma_.lower(),
            candidate_lemma,
            target_token.pos_,
        )

    def _candidate_surface_in_context(self, candidate_word: str, target_token) -> str:
        if target_token.pos_ == "VERB":
            return self.inflect_like_target(candidate_word, target_token)

        return candidate_word

    def _candidate_lemma_in_context(
        self,
        sentence: str,
        target_word: str,
        target_token,
        candidate_word: str,
        target_offset: int | None = None,
    ) -> str | None:
        parse_candidate = self._candidate_surface_in_context(
            candidate_word,
            target_token,
        )

        substituted = self.replace_target(
            sentence,
            target_word,
            parse_candidate,
            target_offset,
        )
        doc = self.nlp(substituted)
        candidate_token = self._find_target_token(doc, parse_candidate, target_offset)

        if candidate_token is None:
            return None

        return candidate_token.lemma_.lower()

    def _nli_contradiction_scores(self, logits: torch.Tensor) -> List[float]:
        if logits.shape[-1] == 1:
            return torch.sigmoid(logits[:, 0]).detach().cpu().tolist()

        contradiction_index = self._nli_contradiction_label_index(logits.shape[-1])
        probs = torch.softmax(logits, dim=-1)
        return probs[:, contradiction_index].detach().cpu().tolist()

    def _nli_contradiction_label_index(self, num_labels: int) -> int:
        id2label = getattr(self.nli_model.config, "id2label", {})

        for index, label in id2label.items():
            if str(label).lower() in {"contradiction", "contradictory", "label_0"}:
                return int(index)

        return 0 if num_labels > 1 else 0

    def _classifier_positive_scores(
        self,
        logits: torch.Tensor,
        model,
        positive_label: str | int | None,
    ) -> List[float]:
        if logits.shape[-1] == 1:
            if getattr(model.config, "problem_type", None) == "regression":
                return logits[:, 0].clamp(0.0, 1.0).detach().cpu().tolist()

            return torch.sigmoid(logits[:, 0]).detach().cpu().tolist()

        positive_index = self._classifier_positive_label_index(
            model,
            positive_label,
            logits.shape[-1],
        )
        probs = torch.softmax(logits, dim=-1)
        return probs[:, positive_index].detach().cpu().tolist()

    def _classifier_positive_label_index(
        self,
        model,
        positive_label: str | int | None,
        num_labels: int,
    ) -> int:
        if isinstance(positive_label, int):
            return positive_label

        if isinstance(positive_label, str):
            label = positive_label.lower()

            if label.isdigit():
                return int(label)

            id2label = getattr(model.config, "id2label", {})
            for index, name in id2label.items():
                if str(name).lower() == label:
                    return int(index)

        id2label = getattr(model.config, "id2label", {})
        positive_label_names = ("true", "valid", "same", "positive", "yes", "label_1", "1")

        for index, name in id2label.items():
            if str(name).lower() in positive_label_names:
                return int(index)

        return 1 if num_labels > 1 else 0


    # BERT MLM fluency scoring

    def mlm_score(
        self,
        sentence: str,
        target_word: str,
        candidates: List[SubstituteCandidate],
        target_offset: int | None = None,
    ) -> List[SubstituteCandidate]:
        masked_sentence = self.replace_target(
            sentence,
            target_word,
            self.bert_tokenizer.mask_token,
            target_offset,
        )

        inputs = self.bert_tokenizer(masked_sentence, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.bert_mlm(**inputs)
            logits = outputs.logits

        mask_index = torch.where(
            inputs["input_ids"][0] == self.bert_tokenizer.mask_token_id
        )[0]

        if len(mask_index) == 0:
            return candidates

        logits = logits[0, mask_index[0]]

        for candidate in candidates:
            token_ids = self.bert_tokenizer.encode(candidate.word, add_special_tokens=False)

            if len(token_ids) == 1:
                token_id = token_ids[0]
                rank = int((logits > logits[token_id]).sum().item()) + 1
                candidate.mlm_score = 1.0 / torch.log2(
                    torch.tensor(rank + 1.0, device=logits.device)
                ).item()
            else:
                candidate.mlm_score = 0.0

        return candidates

    def plausibility_filter(
        self,
        candidates: List[SubstituteCandidate],
        mlm_threshold: float | None = None,
    ) -> List[SubstituteCandidate]:
        mlm_threshold = self.ranker_config.mlm_threshold if mlm_threshold is None else mlm_threshold
        kept = []
        semantic_floor = min(0.45, self.ranker_config.semantic_threshold)

        for candidate in candidates:
            if candidate.lexical_score >= 1.0:
                kept.append(candidate)
                continue

            if candidate.semantic_score < semantic_floor:
                continue

            if self._has_substitute_model() and candidate.substitute_score < min(
                0.05,
                self._candidate_substitute_min_score(candidate),
            ):
                continue

            if (
                candidate.target_score >= self.ranker_config.target_threshold
                or candidate.mlm_score > max(mlm_threshold, 0.05)
                or (
                    self._has_substitute_model()
                    and candidate.substitute_score
                    >= self._candidate_substitute_min_score(candidate)
                )
                or candidate.semantic_score >= self.ranker_config.semantic_threshold
            ):
                kept.append(candidate)

        return kept

    def _candidate_substitute_min_score(self, candidate: SubstituteCandidate) -> float:
        sources = self.candidate_sources.get(candidate.word.lower(), set())
        strong_sources = {"swords", "wordnet", "t5", "t5_balanced", "given"}

        if sources and not (sources & strong_sources):
            return self.ranker_config.weak_source_substitute_min_score

        return self.ranker_config.substitute_min_score

    def _has_substitute_model(self) -> bool:
        return (
            getattr(self, "substitute_model", None) is not None
            and getattr(self, "substitute_tokenizer", None) is not None
        )
