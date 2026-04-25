import torch
from nltk.corpus import wordnet as wn

from .types import SubstituteCandidate


class LinguisticFilterMixin:
    # linguistic filtering with spaCy

    SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"}
    OBJECT_DEPS = {"dobj", "obj"}
    INDIRECT_OBJECT_DEPS = {"dative", "iobj"}
    CLAUSAL_COMPLEMENT_DEPS = {"ccomp", "xcomp"}
    PREDICATE_COMPLEMENT_DEPS = {"acomp", "attr", "oprd"}

    def linguistic_filter(
        self,
        sentence,
        target_word,
        candidates,
        target_offset: int | None = None,
    ):
        doc = self.nlp(sentence)
        target_token = self._find_target_token(doc, target_word, target_offset)

        if target_token is None:
            raise ValueError(f"Target word '{target_word}' not found.")

        sentence_words = {token.text.lower() for token in doc}
        target_lemma = target_token.lemma_.lower()
        target_pos = target_token.pos_
        target_synsets = self._select_target_synsets(sentence, target_lemma, target_pos)
        wordnet_candidates = self._wordnet_candidate_set(target_synsets)

        filtered = []
        seen_candidate_lemmas = set()

        for candidate in candidates:
            candidate = candidate.lower().strip()

            if not self._is_valid_single_word(candidate):
                continue

            if candidate in sentence_words:
                continue

            if candidate in self.nlp.Defaults.stop_words:
                continue

            parse_candidate = candidate
            if target_pos == "VERB":
                parse_candidate = self.inflect_like_target(candidate, target_token)

            substituted = self.replace_target(
                sentence,
                target_word,
                parse_candidate,
                target_offset,
            )
            cand_doc = self.nlp(substituted)
            cand_token = self._find_target_token(
                cand_doc,
                parse_candidate,
                target_offset,
            )

            if cand_token is None:
                continue

            if cand_token.ent_type_ or cand_token.pos_ == "PROPN":
                continue

            if cand_token.is_stop or not cand_token.is_alpha:
                continue

            cand_lemma = cand_token.lemma_.lower()

            if cand_lemma == target_lemma:
                continue

            if target_lemma in cand_lemma or cand_lemma in target_lemma:
                continue

            if self._are_wordnet_opposites(target_lemma, cand_lemma, target_pos):
                continue

            if cand_lemma in seen_candidate_lemmas:
                continue

            if cand_token.pos_ != target_pos:
                continue

            if target_pos == "VERB" and not self._verb_frame_compatible(
                target_token,
                cand_token,
            ):
                continue

            lexical_score = self._lexical_score(
                target_lemma,
                candidate,
                cand_lemma,
                target_pos,
                wordnet_candidates,
            )

            if lexical_score <= 0.0:
                continue

            target_score = 1.0
            if lexical_score < 1.0:
                target_score = self._target_word_similarity(target_lemma, candidate)

            seen_candidate_lemmas.add(cand_lemma)
            filtered.append(
                SubstituteCandidate(
                    word=candidate,
                    pos_match=True,
                    lexical_score=lexical_score,
                    target_score=target_score,
                )
            )

        return filtered

    def _find_target_token(
        self,
        doc,
        target_word: str,
        target_offset: int | None = None,
    ):
        target_word = target_word.lower()

        for token in doc:
            if target_offset is not None:
                if token.idx == target_offset:
                    return token
                continue

            if token.text.lower() == target_word:
                return token

        return None

    def _spacy_pos_to_wordnet(self, pos: str):
        return {
            "ADJ": wn.ADJ,
            "ADV": wn.ADV,
            "NOUN": wn.NOUN,
            "VERB": wn.VERB,
        }.get(pos)

    def _select_target_synsets(
        self,
        sentence: str,
        target_lemma: str,
        target_pos: str,
        max_synsets: int = 1,
    ):
        wn_pos = self._spacy_pos_to_wordnet(target_pos)
        synsets = wn.synsets(target_lemma, pos=wn_pos)

        if len(synsets) <= max_synsets:
            return synsets

        glosses = [
            synset.definition() + " " + " ".join(synset.examples())
            for synset in synsets
        ]
        embeddings = self.sbert.encode([sentence] + glosses, convert_to_tensor=True)
        scores = torch.nn.functional.cosine_similarity(
            embeddings[0],
            embeddings[1:],
            dim=1,
        )
        ranked_indexes = torch.argsort(scores, descending=True)
        return [synsets[index] for index in ranked_indexes[:max_synsets].tolist()]

    def _wordnet_candidate_set(self, target_synsets) -> set[str]:
        candidates = set()

        for synset in target_synsets:
            for lemma in synset.lemmas():
                candidate = lemma.name().replace("_", " ").lower()

                if self._is_valid_single_word(candidate):
                    candidates.add(candidate)

        return candidates

    def _has_wordnet_pos(self, candidate: str, target_pos: str) -> bool:
        wn_pos = self._spacy_pos_to_wordnet(target_pos)
        return bool(wn.synsets(candidate, pos=wn_pos))

    def _lexical_score(
        self,
        target_lemma: str,
        candidate: str,
        candidate_lemma: str,
        target_pos: str,
        wordnet_candidates: set[str],
    ) -> float:
        if candidate in wordnet_candidates or candidate_lemma in wordnet_candidates:
            return 1.0

        sources = self.candidate_sources.get(candidate, set())

        if (
            ("swords" in sources or "wordnet_related" in sources)
            and self._has_wordnet_pos(candidate_lemma, target_pos)
            and candidate_lemma != target_lemma
        ):
            return 0.65

        if (
            "mlm" in sources
            and self._has_wordnet_pos(candidate_lemma, target_pos)
            and candidate_lemma != target_lemma
        ):
            return 0.45

        if (
            "t5" in sources
            and self._has_wordnet_pos(candidate_lemma, target_pos)
            and candidate_lemma != target_lemma
        ):
            return 0.25

        if (
            "given" in sources
            and self._has_wordnet_pos(candidate_lemma, target_pos)
            and candidate_lemma != target_lemma
        ):
            return 0.25

        return 0.0

    def _target_word_similarity(self, target_word: str, candidate: str) -> float:
        embeddings = self.sbert.encode(
            [target_word, candidate],
            convert_to_tensor=True,
        )
        return torch.nn.functional.cosine_similarity(
            embeddings[0],
            embeddings[1],
            dim=0,
        ).item()

    def _are_wordnet_opposites(
        self,
        target_lemma: str,
        candidate_lemma: str,
        target_pos: str,
    ) -> bool:
        wn_pos = self._spacy_pos_to_wordnet(target_pos)

        if wn_pos is None:
            return False

        candidate_forms = self._wordnet_related_forms(candidate_lemma, wn_pos)
        target_opposites = self._wordnet_antonym_forms(target_lemma, wn_pos)

        if candidate_forms & target_opposites:
            return True

        target_forms = self._wordnet_related_forms(target_lemma, wn_pos)
        candidate_opposites = self._wordnet_antonym_forms(candidate_lemma, wn_pos)
        return bool(target_forms & candidate_opposites)

    def _wordnet_antonym_forms(self, lemma: str, wn_pos) -> set[str]:
        forms = set()

        for synset in wn.synsets(lemma, pos=wn_pos):
            for synset_lemma in synset.lemmas():
                for antonym in synset_lemma.antonyms():
                    name = antonym.name().replace("_", " ").lower()

                    if self._is_valid_single_word(name):
                        forms.add(name)

                    for antonym_synset in wn.synsets(name, pos=wn_pos):
                        for antonym_lemma in antonym_synset.lemmas():
                            antonym_name = (
                                antonym_lemma.name().replace("_", " ").lower()
                            )

                            if self._is_valid_single_word(antonym_name):
                                forms.add(antonym_name)

                        for related_synset in antonym_synset.similar_tos():
                            for related_lemma in related_synset.lemmas():
                                related_name = (
                                    related_lemma.name().replace("_", " ").lower()
                                )

                                if self._is_valid_single_word(related_name):
                                    forms.add(related_name)

                    for related_synset in antonym.synset().similar_tos():
                        for related_lemma in related_synset.lemmas():
                            related_name = (
                                related_lemma.name().replace("_", " ").lower()
                            )

                            if self._is_valid_single_word(related_name):
                                forms.add(related_name)

        return forms

    def _wordnet_related_forms(self, lemma: str, wn_pos) -> set[str]:
        forms = {lemma}

        for synset in wn.synsets(lemma, pos=wn_pos):
            for synset_lemma in synset.lemmas():
                name = synset_lemma.name().replace("_", " ").lower()

                if self._is_valid_single_word(name):
                    forms.add(name)

            for related_synset in synset.similar_tos():
                for related_lemma in related_synset.lemmas():
                    name = related_lemma.name().replace("_", " ").lower()

                    if self._is_valid_single_word(name):
                        forms.add(name)

        return forms

    def _verb_frame_compatible(self, target_token, candidate_token) -> bool:
        target_frame = self._verb_argument_frame(target_token)
        candidate_frame = self._verb_argument_frame(candidate_token)

        if not target_frame.issubset(candidate_frame):
            return False

        if {"object", "indirect_object"}.issubset(target_frame):
            return self._has_wordnet_double_object_frame(candidate_token.lemma_.lower())

        return True

    def _verb_argument_frame(self, token) -> set[str]:
        frame = set()

        for child in token.children:
            dep = child.dep_

            if dep in self.SUBJECT_DEPS:
                frame.add("subject")
            elif dep in self.OBJECT_DEPS:
                frame.add("object")
            elif dep in self.INDIRECT_OBJECT_DEPS:
                frame.add("indirect_object")
            elif dep in self.CLAUSAL_COMPLEMENT_DEPS:
                frame.add(dep)
            elif dep in self.PREDICATE_COMPLEMENT_DEPS:
                frame.add("predicate_complement")
            elif dep == "prep":
                frame.add(f"prep:{child.lemma_.lower()}")
            elif dep == "prt":
                frame.add(f"prt:{child.lemma_.lower()}")

        return frame

    def _has_wordnet_double_object_frame(self, candidate_lemma: str) -> bool:
        for synset in wn.synsets(candidate_lemma, pos=wn.VERB):
            for lemma in synset.lemmas():
                if lemma.name().lower().replace("_", " ") != candidate_lemma:
                    continue

                for frame in lemma.frame_strings():
                    if self._is_double_object_wordnet_frame(frame, candidate_lemma):
                        return True

        return False

    def _is_double_object_wordnet_frame(
        self,
        frame: str,
        candidate_lemma: str,
    ) -> bool:
        frame = frame.lower()
        candidate_lemma = candidate_lemma.lower()
        verb_forms = {candidate_lemma, f"{candidate_lemma}s", f"{candidate_lemma}ing"}

        for verb_form in verb_forms:
            pattern = f"somebody {verb_form} somebody something"

            if pattern in frame:
                return True

        return False
