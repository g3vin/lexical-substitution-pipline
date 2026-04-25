import argparse
import json
import os
from importlib import resources
from pathlib import Path

# disable the annoying bert warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from . import LexicalSubstitutionPipeline, RankerConfig

try:
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
except ImportError:
    pass


DEFAULT_SENTENCE = "The dog chased the cat."
DEFAULT_TARGET = "chased"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _first_existing_path(candidates: list[Path]) -> str | None:
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def default_substitute_model_path() -> str | None:
    return _first_existing_path(
        [
            Path.cwd() / "substitute-roberta",
            PROJECT_ROOT / "substitute-roberta",
        ]
    )


def default_swords_path() -> str | None:
    return _first_existing_path(
        [
            Path.cwd() / "data" / "swords" / "swords-v1.1_dev.json.gz",
            PROJECT_ROOT / "data" / "swords" / "swords-v1.1_dev.json.gz",
        ]
    )


def load_bundled_ranker_config() -> RankerConfig:
    with resources.files("pipeline").joinpath("resources/default_ranker_config.json").open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    valid_keys = set(RankerConfig.__dataclass_fields__)
    return RankerConfig(**{key: value for key, value in data.items() if key in valid_keys})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ranked lexical substitutes for a target word in context."
    )
    parser.add_argument(
        "sentence",
        nargs="?",
        default=DEFAULT_SENTENCE,
        help="Input sentence containing the target word.",
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=DEFAULT_TARGET,
        help="Target word to replace.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--t5-model",
        default="google/flan-t5-large",
        help="T5/FLAN model used for LLM candidate generation.",
    )
    parser.add_argument(
        "--generation-mode",
        default="full",
        choices=["full", "balanced", "fast", "resources", "mlm", "t5"],
        help=(
            "Candidate generation mode. balanced uses resources, compact T5 prompts, "
            "and MLM; fast uses resources and MLM."
        ),
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Disable cross-encoder reranking for faster runs.",
    )
    parser.add_argument(
        "--nli-model",
        default=None,
        help=(
            "Optional NLI model/path used to filter contradictory substitutions. "
            "Disabled by default because it adds model load/runtime cost."
        ),
    )
    parser.add_argument(
        "--nli-contradiction-threshold",
        type=float,
        default=0.80,
        help="Contradiction probability threshold for --nli-model filtering.",
    )
    parser.add_argument(
        "--target-offset",
        type=int,
        default=None,
        help="Optional character offset for the target if it appears multiple times.",
    )
    parser.add_argument(
        "--substitute-model",
        default=default_substitute_model_path(),
        help=(
            "Path to a trained substitute regression/classification model. "
            "Auto-detects ./substitute-roberta when available."
        ),
    )
    parser.add_argument(
        "--no-substitute-model",
        action="store_true",
        help="Disable loading the substitute model even if it exists.",
    )
    parser.add_argument(
        "--substitute-threshold",
        type=float,
        default=0.5,
        help="Threshold used only when --substitute-hard-filter is enabled.",
    )
    parser.add_argument(
        "--substitute-hard-filter",
        action="store_true",
        help="Use the substitute model as a hard filter instead of a soft score.",
    )
    parser.add_argument(
        "--ranker-config",
        default=None,
        help="Optional path to a ranker config JSON file. Uses the bundled default when omitted.",
    )
    parser.add_argument(
        "--learned-ranker-model",
        default=None,
        help="Optional path to a trained learned ranker artifact.",
    )
    parser.add_argument(
        "--no-ranker-config",
        action="store_true",
        help="Disable loading any ranker config and use code defaults instead.",
    )
    parser.add_argument(
        "--no-morphology",
        action="store_true",
        help="Disable simple morphology preservation.",
    )
    parser.add_argument(
        "--swords-path",
        default=default_swords_path(),
        help=(
            "Optional local SWORDS JSON.GZ path for lexical-resource candidate recall. "
            "Auto-detects ./data/swords/swords-v1.1_dev.json.gz when available."
        ),
    )
    parser.add_argument(
        "--no-swords",
        action="store_true",
        help="Disable SWORDS lexical-resource candidates.",
    )
    parser.add_argument(
        "--swords-min-score",
        type=float,
        default=0.5,
        help="Minimum SWORDS label score for lexical-resource candidates.",
    )
    parser.add_argument(
        "--swords-max-candidates",
        type=int,
        default=30,
        help="Maximum SWORDS candidates to add for the target lemma/POS.",
    )
    parser.add_argument(
        "--wordnet-max-synsets",
        type=int,
        default=3,
        help="Number of context-ranked WordNet synsets to mine for lexical candidates.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print only ranked words, one per line.",
    )
    return parser.parse_args()


def build_pipeline(args):
    pipeline_kwargs = {
        "t5_model_name": args.t5_model,
        "generation_mode": args.generation_mode,
        "enable_reranker": not args.no_reranker,
        "nli_model_name": args.nli_model,
        "nli_contradiction_threshold": args.nli_contradiction_threshold,
        "swords_path": None if args.no_swords else args.swords_path,
        "swords_min_score": args.swords_min_score,
        "swords_max_candidates": args.swords_max_candidates,
        "wordnet_max_synsets": args.wordnet_max_synsets,
    }

    if not args.no_substitute_model and args.substitute_model is not None:
        substitute_model_path = Path(args.substitute_model)
        if substitute_model_path.exists():
            pipeline_kwargs.update(
                {
                    "substitute_classifier_model_name": str(substitute_model_path),
                    "substitute_positive_label": "valid",
                    "substitute_threshold": args.substitute_threshold,
                    "substitute_hard_filter": args.substitute_hard_filter,
                }
            )

    if not args.no_ranker_config:
        if args.ranker_config is not None:
            ranker_config_path = Path(args.ranker_config)
            if ranker_config_path.exists():
                pipeline_kwargs["ranker_config_path"] = str(ranker_config_path)
        else:
            pipeline_kwargs["ranker_config"] = load_bundled_ranker_config()

    if args.learned_ranker_model is not None:
        learned_ranker_path = Path(args.learned_ranker_model)
        if learned_ranker_path.exists():
            pipeline_kwargs["learned_ranker_model_path"] = str(learned_ranker_path)

    pipeline_kwargs["preserve_morphology"] = not args.no_morphology
    return LexicalSubstitutionPipeline(**pipeline_kwargs)


def print_results(results, compact: bool):
    if not results:
        print("No substitutes survived the filters.")
        return

    for rank, candidate in enumerate(results, start=1):
        if compact:
            print(candidate.word)
            continue

        print(
            f"{rank}. {candidate.word} "
            f"| lexical={candidate.lexical_score:.4f} "
            f"| target={candidate.target_score:.4f} "
            f"| semantic={candidate.semantic_score:.4f} "
            f"| mlm_rank={candidate.mlm_score:.4f} "
            f"| substitute={candidate.substitute_score:.4f} "
            f"| rerank={candidate.rerank_score:.4f} "
            f"| final={candidate.final_score:.4f}"
        )


def main():
    args = parse_args()
    pipeline = build_pipeline(args)
    results = pipeline.substitute(
        args.sentence,
        args.target,
        top_k=args.top_k,
        target_offset=args.target_offset,
    )
    print_results(results, args.compact)
