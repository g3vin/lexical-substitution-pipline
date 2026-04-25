import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from evaluate_swords import apply_config, score_swords, summarize
from pipeline import LexicalSubstitutionPipeline, RankerConfig
from pipeline.learned_ranker import LearnedRanker
from swords_utils import load_or_download_swords


FEATURE_COLUMNS = ["lexical", "target", "semantic", "mlm", "substitute", "rerank"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a learned lexical substitution ranker on grouped SWORDS candidates."
    )
    parser.add_argument("--features-csv", default=None)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--data-dir", default="data/swords")
    parser.add_argument("--swords-path", default=None)
    parser.add_argument("--ranker-config", default="ranker_config.json")
    parser.add_argument("--substitute-model", default="./substitute-roberta")
    parser.add_argument("--substitute-threshold", type=float, default=0.5)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--acceptable-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-model", default="learned_ranker.pkl")
    return parser.parse_args()


def feature_results_from_dataframe(frame: pd.DataFrame):
    rows = []

    for target_id, group in frame.groupby("target_id", sort=False):
        rows.append(
            {
                "target_id": target_id,
                "target": group["target"].iloc[0],
                "gold_scores": group["gold_score"].tolist(),
                "features": group[FEATURE_COLUMNS].to_dict("records"),
            }
        )

    return rows


def load_training_frame(args) -> pd.DataFrame:
    if args.features_csv is not None:
        return pd.read_csv(args.features_csv)

    swords, _ = load_or_download_swords(
        split=args.split,
        data_dir=args.data_dir,
        path=args.swords_path,
    )
    pipeline = LexicalSubstitutionPipeline(
        ranker_config_path=args.ranker_config,
        substitute_classifier_model_name=args.substitute_model,
        substitute_positive_label="valid",
        substitute_threshold=args.substitute_threshold,
    )
    feature_results = score_swords(pipeline, swords, max_targets=args.max_targets)
    flat_rows = []

    for result in feature_results:
        for gold_score, feature_row in zip(result["gold_scores"], result["features"]):
            flat_rows.append(
                {
                    "target_id": result["target_id"],
                    "target": result["target"],
                    "gold_score": gold_score,
                    **feature_row,
                }
            )

    return pd.DataFrame(flat_rows)


def load_ranker_config(path: str | None) -> RankerConfig:
    if path is None:
        return RankerConfig()

    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    valid_keys = set(RankerConfig.__dataclass_fields__)
    return RankerConfig(**{key: value for key, value in payload.items() if key in valid_keys})


def main():
    args = parse_args()
    frame = load_training_frame(args)
    groups = frame["target_id"]
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.validation_fraction,
        random_state=args.seed,
    )
    train_index, validation_index = next(splitter.split(frame, groups=groups))
    train_frame = frame.iloc[train_index].reset_index(drop=True)
    validation_frame = frame.iloc[validation_index].reset_index(drop=True)

    ranker = LearnedRanker()
    ranker.fit(
        train_frame[FEATURE_COLUMNS].to_dict("records"),
        train_frame["gold_score"].tolist(),
    )

    config = load_ranker_config(args.ranker_config)
    validation_results = feature_results_from_dataframe(validation_frame)
    metrics = summarize(
        apply_config(validation_results, config, learned_ranker=ranker),
        args.top_k,
        args.acceptable_threshold,
    )

    ranker.metadata = {
        "top_k": args.top_k,
        "acceptable_threshold": args.acceptable_threshold,
        "validation_fraction": args.validation_fraction,
        "train_targets": int(train_frame["target_id"].nunique()),
        "validation_targets": int(validation_frame["target_id"].nunique()),
        "metrics": metrics,
    }
    ranker.save(args.output_model)

    print(
        json.dumps(
            {
                "output_model": args.output_model,
                "rows": len(frame),
                "train_rows": len(train_frame),
                "validation_rows": len(validation_frame),
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
