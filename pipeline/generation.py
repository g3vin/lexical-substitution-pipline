from collections import defaultdict
from typing import List
import re

import torch
from nltk.corpus import wordnet as wn


class CandidateGenerationMixin:
    # candidate generation

    def generate_candidates(
        self,
        sentence: str,
        target_word: str,
        num_candidates: int = 60,
    ) -> List[str]:
        self.candidate_sources = defaultdict(set)

        candidates = []
        resource_candidates = self.generate_lexical_resource_candidates(
            sentence,
            target_word,
        )
        candidates.extend(resource_candidates)

        if self.generation_mode == "balanced":
            candidates.extend(
                self.generate_balanced_t5_candidates(
                    sentence,
                    target_word,
                    resource_candidates=resource_candidates,
                )
            )
        elif self.generation_mode in {"full", "t5"}:
            candidates.extend(
                self.generate_t5_candidates(
                    sentence,
                    target_word,
                    num_candidates,
                    resource_candidates=resource_candidates,
                )
            )

        if self.generation_mode in {"full", "balanced", "fast", "mlm"}:
            candidates.extend(self.generate_mlm_candidates(sentence, target_word, 150))

        return list(dict.fromkeys(candidates))

    def generate_t5_candidates(
        self,
        sentence: str,
        target_word: str,
        num_candidates: int,
        resource_candidates: List[str] | None = None,
    ) -> List[str]:
        if self.t5_model is None or self.t5_tokenizer is None:
            return []

        candidates = []
        resource_hint = self.lexical_resource_prompt(resource_candidates or [])

        prompts = [
            f"synonyms of {target_word}:",
            f"similar words to {target_word}:",
            f"single-word alternatives for {target_word}:",
            (
                f"Sentence: {sentence}\n"
                f"Target word: {target_word}\n"
                f"{resource_hint}\n"
                "Using the lexical resources as hints, generate context-appropriate "
                "single-word substitutes. Return only comma-separated base-form words."
            ),
            (
                f"List 20 comma-separated single-word synonyms for '{target_word}' "
                f"as used in this sentence: {sentence} {resource_hint}"
            ),
            (
                f"In the sentence '{sentence}', replace '{target_word}' with "
                "20 possible single-word alternatives. "
                f"{resource_hint} Output only comma-separated words."
            ),
            (
                f"Single-word synonyms for '{target_word}' in context: {sentence}. "
                f"{resource_hint} Return a comma-separated list."
            ),
            f"replace {target_word} with one synonym in: {sentence}",
        ]

        sequences_per_prompt = max(1, num_candidates // len(prompts))

        for prompt in prompts:
            inputs = self.t5_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.t5_model.generate(
                    **inputs,
                    max_new_tokens=64,
                    num_beams=1,
                    num_return_sequences=sequences_per_prompt,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=1.0,
                    repetition_penalty=1.2,
                )

            for output in outputs:
                text = self.t5_tokenizer.decode(output, skip_special_tokens=True)
                pieces = re.findall(r"[a-zA-Z]+", text)

                for piece in pieces:
                    piece = piece.strip().lower()

                    if self._is_valid_single_word(piece):
                        self.candidate_sources[piece].add("t5")
                        candidates.append(piece)

        return candidates

    def generate_balanced_t5_candidates(
        self,
        sentence: str,
        target_word: str,
        resource_candidates: List[str] | None = None,
    ) -> List[str]:
        if self.t5_model is None or self.t5_tokenizer is None:
            return []

        resource_hint = self.lexical_resource_prompt(resource_candidates or [])
        prompts = [
            (
                f"Sentence: {sentence}\n"
                f"Target word: {target_word}\n"
                f"{resource_hint}\n"
                "Generate context-appropriate single-word substitutes for the target. "
                "Use the lexical resources as hints, but only include words that fit "
                "the sentence. Return only comma-separated base-form words."
            ),
            (
                f"In this sentence, replace only '{target_word}' with natural "
                f"single-word alternatives: {sentence}\n"
                "Return comma-separated base-form words."
            ),
            (
                f"Literary sentence: {sentence}\n"
                f"Target: {target_word}\n"
                "List precise, possibly literary, one-word substitutes that preserve "
                "the meaning. Return comma-separated base-form words."
            ),
        ]

        candidates = []

        for prompt in prompts:
            inputs = self.t5_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.t5_model.generate(
                    **inputs,
                    max_new_tokens=40,
                    num_beams=1,
                    num_return_sequences=1,
                    do_sample=True,
                    top_k=40,
                    top_p=0.90,
                    temperature=0.8,
                    repetition_penalty=1.15,
                )

            for output in outputs:
                text = self.t5_tokenizer.decode(output, skip_special_tokens=True)
                pieces = re.findall(r"[a-zA-Z]+", text)

                for piece in pieces:
                    piece = piece.strip().lower()

                    if self._is_valid_single_word(piece):
                        self.candidate_sources[piece].add("t5_balanced")
                        candidates.append(piece)

        return candidates

    def generate_wordnet_candidates(self, sentence: str, target_word: str) -> List[str]:
        doc = self.nlp(sentence)
        target_token = self._find_target_token(doc, target_word)

        if target_token is None:
            return []

        candidates = []

        for candidate in self.generate_expanded_wordnet_candidates(
            sentence,
            target_token.lemma_.lower(),
            target_token.pos_,
        ):
            candidates.append(candidate)

        return candidates

    def generate_mlm_candidates(
        self,
        sentence: str,
        target_word: str,
        num_candidates: int = 150,
    ) -> List[str]:
        if self.bert_mlm is None or self.bert_tokenizer is None:
            return []

        masked_sentence = self.replace_target(
            sentence,
            target_word,
            self.bert_tokenizer.mask_token,
        )

        inputs = self.bert_tokenizer(masked_sentence, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.bert_mlm(**inputs)

        mask_index = torch.where(
            inputs["input_ids"][0] == self.bert_tokenizer.mask_token_id
        )[0]

        if len(mask_index) == 0:
            return []

        logits = outputs.logits[0, mask_index[0]]
        top_token_ids = torch.topk(logits, k=num_candidates).indices
        candidates = []

        for token_id in top_token_ids:
            candidate = self.bert_tokenizer.decode([token_id]).strip().lower()

            if candidate.startswith("##"):
                continue

            if self._is_valid_single_word(candidate):
                self.candidate_sources[candidate].add("mlm")
                candidates.append(candidate)

        return candidates

    def _is_valid_single_word(self, word: str) -> bool:
        return bool(re.fullmatch(r"[a-zA-Z]+", word))
