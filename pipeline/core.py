from collections import defaultdict
import json

import spacy
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from .filters import SemanticAndClassifierFilterMixin
from .generation import CandidateGenerationMixin
from .learned_ranker import load_learned_ranker
from .lexical_resources import LexicalResourceMixin
from .linguistic import LinguisticFilterMixin
from .morphology import MorphologyMixin
from .ranking import RankingMixin
from .types import RankerConfig
from .workflow import WorkflowMixin


class LexicalSubstitutionPipeline(
    CandidateGenerationMixin,
    LexicalResourceMixin,
    LinguisticFilterMixin,
    SemanticAndClassifierFilterMixin,
    RankingMixin,
    WorkflowMixin,
    MorphologyMixin,
):
    def __init__(
        self,
        t5_model_name: str = "google/flan-t5-base",
        sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        bert_model_name: str = "bert-base-uncased",
        wic_model_name: str | None = None,
        wic_positive_label: str | int | None = None,
        wic_threshold: float = 0.5,
        nli_model_name: str | None = None,
        nli_contradiction_threshold: float = 0.80,
        substitute_classifier_model_name: str | None = None,
        substitute_positive_label: str | int | None = None,
        substitute_threshold: float = 0.5,
        substitute_hard_filter: bool = False,
        learned_ranker_model_path: str | None = None,
        ranker_config_path: str | None = None,
        ranker_config: RankerConfig | None = None,
        preserve_morphology: bool = True,
        cross_encoder_name: str = "cross-encoder/stsb-roberta-base",
        generation_mode: str = "full",
        enable_reranker: bool = True,
        swords_path: str | None = "data/swords/swords-v1.1_dev.json.gz",
        swords_min_score: float = 0.5,
        swords_max_candidates: int = 30,
        wordnet_max_synsets: int = 3,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.wic_positive_label = wic_positive_label
        self.wic_threshold = wic_threshold
        self.nli_contradiction_threshold = nli_contradiction_threshold
        self.substitute_positive_label = substitute_positive_label
        self.substitute_threshold = substitute_threshold
        self.substitute_hard_filter = substitute_hard_filter
        self.learned_ranker = None
        self.ranker_config = ranker_config or RankerConfig()
        self.preserve_morphology = preserve_morphology
        self.candidate_sources = defaultdict(set)
        self.generation_mode = generation_mode
        self.enable_reranker = enable_reranker

        valid_generation_modes = {"full", "balanced", "fast", "resources", "mlm", "t5"}
        if self.generation_mode not in valid_generation_modes:
            raise ValueError(
                "generation_mode must be one of: "
                + ", ".join(sorted(valid_generation_modes))
            )

        if ranker_config_path is not None:
            self.load_ranker_config(ranker_config_path)
        elif substitute_classifier_model_name is not None and ranker_config is None:
            self.ranker_config = RankerConfig(
                lexical_weight=0.20,
                target_weight=0.10,
                semantic_weight=0.15,
                mlm_weight=0.20,
                substitute_weight=0.30,
                rerank_weight=0.05,
                semantic_threshold=self.ranker_config.semantic_threshold,
                target_threshold=self.ranker_config.target_threshold,
                mlm_threshold=self.ranker_config.mlm_threshold,
            )

        if learned_ranker_model_path is not None:
            self.learned_ranker = load_learned_ranker(learned_ranker_model_path)

        self.nlp = spacy.load("en_core_web_sm")
        self.configure_lexical_resources(
            swords_path=swords_path,
            swords_min_score=swords_min_score,
            swords_max_candidates=swords_max_candidates,
            wordnet_max_synsets=wordnet_max_synsets,
        )

        self.t5_tokenizer = None
        self.t5_model = None
        if self.generation_mode in {"full", "balanced", "t5"}:
            self.t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
            self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name).to(
                self.device
            )

        self.sbert = SentenceTransformer(sbert_model_name, device=self.device)

        self.bert_tokenizer = None
        self.bert_mlm = None
        if self.generation_mode in {"full", "balanced", "fast", "mlm"}:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            self.bert_mlm = AutoModelForMaskedLM.from_pretrained(bert_model_name).to(
                self.device
            )

        self.wic_tokenizer = None
        self.wic_model = None
        if wic_model_name is not None:
            self.wic_tokenizer = AutoTokenizer.from_pretrained(wic_model_name)
            self.wic_model = AutoModelForSequenceClassification.from_pretrained(
                wic_model_name
            ).to(self.device)
            self.wic_model.eval()

        self.nli_tokenizer = None
        self.nli_model = None
        if nli_model_name is not None:
            self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(
                nli_model_name
            ).to(self.device)
            self.nli_model.eval()

        self.substitute_tokenizer = None
        self.substitute_model = None
        if substitute_classifier_model_name is not None:
            self.substitute_tokenizer = AutoTokenizer.from_pretrained(
                substitute_classifier_model_name
            )
            self.substitute_model = AutoModelForSequenceClassification.from_pretrained(
                substitute_classifier_model_name
            ).to(self.device)
            self.substitute_model.eval()

        self.cross_encoder = None
        if self.enable_reranker:
            self.cross_encoder = CrossEncoder(cross_encoder_name, device=self.device)

    def load_ranker_config(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        valid_keys = set(RankerConfig.__dataclass_fields__)
        config = {
            key: value
            for key, value in data.items()
            if key in valid_keys
        }
        self.ranker_config = RankerConfig(**{**self.ranker_config.__dict__, **config})
