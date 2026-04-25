from __future__ import annotations

from collections import defaultdict
import gzip
import json
from pathlib import Path
from typing import Iterable

from nltk.corpus import wordnet as wn


class LexicalResourceMixin:
    # Candidate recall from lexical resources such as WordNet and SWORDS.

    def configure_lexical_resources(
        self,
        swords_path: str | None = None,
        swords_min_score: float = 0.5,
        swords_max_candidates: int = 30,
        wordnet_max_synsets: int = 3,
    ):
        self.swords_path = swords_path
        self.swords_min_score = swords_min_score
        self.swords_max_candidates = swords_max_candidates
        self.wordnet_max_synsets = wordnet_max_synsets
        self.swords_candidate_index = defaultdict(dict)

        if swords_path is not None and Path(swords_path).exists():
            self.swords_candidate_index = self._load_swords_candidate_index(
                swords_path,
                swords_min_score,
            )

    def generate_lexical_resource_candidates(
        self,
        sentence: str,
        target_word: str,
        target_offset: int | None = None,
    ) -> list[str]:
        doc = self.nlp(sentence)
        target_token = self._find_target_token(doc, target_word, target_offset)

        if target_token is None:
            return []

        target_lemma = target_token.lemma_.lower()
        target_pos = target_token.pos_

        candidates = []
        candidates.extend(
            self.generate_expanded_wordnet_candidates(
                sentence,
                target_lemma,
                target_pos,
            )
        )
        candidates.extend(
            self.generate_swords_candidates(
                target_lemma,
                target_pos,
            )
        )

        return list(dict.fromkeys(candidates))

    def generate_expanded_wordnet_candidates(
        self,
        sentence: str,
        target_lemma: str,
        target_pos: str,
    ) -> list[str]:
        synsets = self._select_target_synsets(
            sentence,
            target_lemma,
            target_pos,
            max_synsets=self.wordnet_max_synsets,
        )
        candidates = []

        for synset in synsets:
            for candidate in self._wordnet_synset_candidates(synset):
                self.candidate_sources[candidate].add("wordnet")
                candidates.append(candidate)

            for related_synset in self._related_wordnet_synsets(synset):
                for candidate in self._wordnet_synset_candidates(related_synset):
                    self.candidate_sources[candidate].add("wordnet_related")
                    candidates.append(candidate)

            for lemma in synset.lemmas():
                for related in lemma.derivationally_related_forms():
                    candidate = related.name().replace("_", " ").lower()

                    if self._is_valid_single_word(candidate):
                        self.candidate_sources[candidate].add("wordnet_related")
                        candidates.append(candidate)

        return list(dict.fromkeys(candidates))

    def generate_swords_candidates(
        self,
        target_lemma: str,
        target_pos: str,
    ) -> list[str]:
        candidates_by_score = self.swords_candidate_index.get((target_lemma, target_pos), {})

        if not candidates_by_score:
            candidates_by_score = self.swords_candidate_index.get((target_lemma, None), {})

        ranked = sorted(
            candidates_by_score.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        candidates = []

        for candidate, score in ranked[: self.swords_max_candidates]:
            if not self._is_valid_single_word(candidate):
                continue

            self.candidate_sources[candidate].add("swords")
            candidates.append(candidate)

        return candidates

    def lexical_resource_prompt(
        self,
        resource_candidates: Iterable[str],
        max_candidates: int = 40,
    ) -> str:
        seeds = list(dict.fromkeys(resource_candidates))[:max_candidates]

        if not seeds:
            return ""

        return "Lexical resource candidates: " + ", ".join(seeds) + "."

    def _wordnet_synset_candidates(self, synset) -> list[str]:
        candidates = []

        for lemma in synset.lemmas():
            candidate = lemma.name().replace("_", " ").lower()

            if self._is_valid_single_word(candidate):
                candidates.append(candidate)

        return candidates

    def _related_wordnet_synsets(self, synset):
        related = []

        for relation in (
            synset.similar_tos,
            synset.verb_groups,
            synset.hypernyms,
            synset.hyponyms,
        ):
            related.extend(relation())

        return related

    def _load_swords_candidate_index(
        self,
        path: str,
        min_score: float,
    ) -> defaultdict[tuple[str, str | None], dict[str, float]]:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            swords = json.load(f)

        target_id_to_substitutes = defaultdict(list)

        for substitute_id, substitute in swords["substitutes"].items():
            target_id_to_substitutes[substitute["target_id"]].append(substitute_id)

        index = defaultdict(dict)

        for target_id, target in swords["targets"].items():
            target_lemma = self._swords_target_lemma(target["target"], target.get("pos"))
            target_pos = target.get("pos")

            for substitute_id in target_id_to_substitutes[target_id]:
                substitute = swords["substitutes"][substitute_id]["substitute"].lower()
                labels = swords["substitute_labels"][substitute_id]
                score = self._swords_label_score(labels)

                if score < min_score or not self._is_valid_single_word(substitute):
                    continue

                keyed_scores = index[(target_lemma, target_pos)]
                keyed_scores[substitute] = max(score, keyed_scores.get(substitute, 0.0))

                fallback_scores = index[(target_lemma, None)]
                fallback_scores[substitute] = max(
                    score,
                    fallback_scores.get(substitute, 0.0),
                )

        return index

    def _swords_target_lemma(self, target: str, pos: str | None) -> str:
        wn_pos = self._spacy_pos_to_wordnet(pos) if pos is not None else None
        synsets = wn.synsets(target, pos=wn_pos)

        if synsets:
            for lemma in synsets[0].lemmas():
                if lemma.name().lower().replace("_", " ") == target.lower():
                    return lemma.name().lower().replace("_", " ")

        doc = self.nlp(target)

        if doc:
            return doc[0].lemma_.lower()

        return target.lower()

    def _swords_label_score(self, labels: list[str]) -> float:
        if not labels:
            return 0.0

        return labels.count("TRUE") / len(labels)
