from typing import List
import re

from lemminflect import getInflection, getLemma

from .types import TARGET_START, TARGET_END, SubstituteCandidate


class MorphologyMixin:
    # utility

    def replace_target(
        self,
        sentence: str,
        target_word: str,
        replacement: str,
        target_offset: int | None = None,
    ) -> str:
        if target_offset is not None:
            end = target_offset + len(target_word)
            return sentence[:target_offset] + replacement + sentence[end:]

        pattern = rf"\b{re.escape(target_word)}\b"
        return re.sub(pattern, replacement, sentence, count=1, flags=re.IGNORECASE)

    def mark_target(
        self,
        sentence: str,
        target_word: str,
        target_offset: int | None = None,
    ) -> str:
        if target_offset is not None:
            end = target_offset + len(target_word)
            return (
                sentence[:target_offset]
                + TARGET_START
                + " "
                + sentence[target_offset:end]
                + " "
                + TARGET_END
                + sentence[end:]
            )

        pattern = rf"\b{re.escape(target_word)}\b"
        return re.sub(
            pattern,
            lambda match: f"<target> {match.group(0)} </target>",
            sentence,
            count=1,
            flags=re.IGNORECASE,
        )

    def apply_morphology(
        self,
        sentence: str,
        target_word: str,
        candidates: List[SubstituteCandidate],
        target_offset: int | None = None,
    ) -> List[SubstituteCandidate]:
        doc = self.nlp(sentence)
        target_token = self._find_target_token(doc, target_word, target_offset)

        if target_token is None:
            return candidates

        kept = []

        for candidate in candidates:
            inflected = self.inflect_like_target(candidate.word, target_token)

            if self._valid_inflected_substitute(
                sentence,
                target_word,
                inflected,
                target_token,
                target_offset,
            ):
                candidate.word = inflected
                kept.append(candidate)

        return kept

    def inflect_like_target(self, candidate: str, target_token) -> str:
        tag = target_token.tag_
        lemma = self._lemma_for_tag(candidate, tag)

        inflected = self._lemminflect(lemma, tag)

        if inflected is not None:
            return self._match_case(inflected, target_token.text)

        if tag in {"NNS", "NNPS"}:
            return self._pluralize(candidate)

        if tag == "VBG":
            return self._to_ing(candidate)

        if tag in {"VBD", "VBN"}:
            return self._to_past(candidate)

        if tag == "VBZ":
            return self._third_person_singular(candidate)

        if tag == "JJR":
            return self._comparative(candidate)

        if tag == "JJS":
            return self._superlative(candidate)

        if target_token.text[:1].isupper():
            return candidate.capitalize()

        return candidate

    def _lemma_for_tag(self, candidate: str, tag: str) -> str:
        if tag.startswith("V"):
            return self._lemma_for_upos(candidate, "VERB")

        if tag in {"NN", "NNS", "NNP", "NNPS"}:
            return self._lemma_for_upos(candidate, "NOUN")

        if tag in {"JJ", "JJR", "JJS"}:
            return self._lemma_for_upos(candidate, "ADJ")

        if tag in {"RB", "RBR", "RBS"}:
            return self._lemma_for_upos(candidate, "ADV")

        return candidate

    def _lemma_for_upos(self, word: str, upos: str) -> str:
        lemmas = getLemma(word, upos=upos)

        if lemmas:
            return lemmas[0]

        return word

    def _valid_inflected_substitute(
        self,
        sentence: str,
        target_word: str,
        substitute: str,
        target_token,
        target_offset: int | None = None,
    ) -> bool:
        if not self._is_valid_single_word(substitute):
            return False

        if substitute.lower() == target_token.text.lower():
            return False

        if self._looks_like_bad_inflection(substitute):
            return False

        substituted = self.replace_target(
            sentence,
            target_word,
            substitute,
            target_offset,
        )
        doc = self.nlp(substituted)
        substitute_token = self._find_target_token(doc, substitute, target_offset)

        if substitute_token is None:
            return False

        if substitute_token.pos_ != target_token.pos_:
            return False

        return True

    def _looks_like_bad_inflection(self, word: str) -> bool:
        lowered = word.lower()

        if re.search(r"(ses|xes|zes|ches|shes)es$", lowered):
            return True

        if lowered.endswith("est") and lowered[:-3].endswith(("er", "est")):
            return True

        if lowered.endswith("er") and lowered[:-2].endswith("er"):
            return True

        return False

    def _lemminflect(self, lemma: str, tag: str) -> str | None:
        forms = getInflection(lemma, tag=tag)

        if forms:
            return forms[0]

        return None

    def _verb_lemma(self, word: str) -> str:
        return self._lemma_for_upos(word, "VERB")

    def _match_case(self, word: str, target_word: str) -> str:
        if target_word.isupper():
            return word.upper()

        if target_word[:1].isupper():
            return word.capitalize()

        return word

    def _pluralize(self, word: str) -> str:
        if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
            return word[:-1] + "ies"
        if word.endswith(("s", "x", "z", "ch", "sh")):
            return word + "es"
        return word + "s"

    def _to_ing(self, word: str) -> str:
        if word.endswith("ie"):
            return word[:-2] + "ying"
        if word.endswith("e") and not word.endswith("ee"):
            return word[:-1] + "ing"
        return word + "ing"

    def _to_past(self, word: str) -> str:
        if word.endswith("e"):
            return word + "d"
        if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
            return word[:-1] + "ied"
        return word + "ed"

    def _third_person_singular(self, word: str) -> str:
        if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
            return word[:-1] + "ies"
        if word.endswith(("s", "x", "z", "ch", "sh")):
            return word + "es"
        return word + "s"

    def _comparative(self, word: str) -> str:
        if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
            return word[:-1] + "ier"
        return word + "er"

    def _superlative(self, word: str) -> str:
        if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
            return word[:-1] + "iest"
        return word + "est"
