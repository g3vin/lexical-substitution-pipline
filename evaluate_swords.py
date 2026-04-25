import argparse
import json
import math
from pathlib import Path

from pipeline import LexicalSubstitutionPipeline, RankerConfig
from pipeline.learned_ranker import load_learned_ranker
from swords_utils import download_swords, iter_swords_targets, load_or_download_swords


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate and optionally autotune the lexical substitution pipeline on SWORDS."
    )
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--data-dir", default="data/swords")
    parser.add_argument("--swords-path", default=None)
    parser.add_argument(
        "--resource-swords-split",
        default=None,
        choices=["dev", "test"],
        help=(
            "SWORDS split used only as a lexical candidate resource. "
            "Defaults to the opposite of --split."
        ),
    )
    parser.add_argument(
        "--resource-swords-path",
        default=None,
        help="Explicit SWORDS JSON.GZ path used only as a lexical candidate resource.",
    )
    parser.add_argument(
        "--no-resource-swords",
        action="store_true",
        help="Disable SWORDS lexical-resource candidates during evaluation.",
    )
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--acceptable-threshold", type=float, default=0.5)
    parser.add_argument("--ranker-config", default=None)
    parser.add_argument("--substitute-model", default=None)
    parser.add_argument(
        "--learned-ranker-model",
        default=None,
        help="Optional path to a trained learned ranker artifact.",
    )
    parser.add_argument("--substitute-threshold", type=float, default=0.5)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--output-config", default="ranker_config.json")
    parser.add_argument("--output-metrics", default=None)
    return parser.parse_args()


def opposite_swords_split(split):
    return "test" if split == "dev" else "dev"


def resolve_resource_swords(args, eval_path):
    if args.no_resource_swords:
        return None, None

    if args.resource_swords_path is not None:
        resource_path = Path(args.resource_swords_path)
        resource_split = args.resource_swords_split or "custom"

        if not resource_path.exists():
            raise FileNotFoundError(
                f"SWORDS resource file does not exist: {resource_path}"
            )
    else:
        resource_split = args.resource_swords_split or opposite_swords_split(args.split)
        resource_path = download_swords(resource_split, args.data_dir)

    if resource_path.resolve() == Path(eval_path).resolve():
        raise ValueError(
            "SWORDS evaluation split and lexical-resource split point to the same file. "
            "Use --resource-swords-split with the other split, "
            "--resource-swords-path with a different file, or --no-resource-swords."
        )

    return str(resource_path), resource_split


def dcg(gains):
    return sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))


def ndcg_at_k(gold_scores, predicted_scores, k):
    order = sorted(range(len(gold_scores)), key=lambda i: predicted_scores[i], reverse=True)
    ideal = sorted(gold_scores, reverse=True)
    actual_dcg = dcg([gold_scores[i] for i in order[:k]])
    ideal_dcg = dcg(ideal[:k])
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def average_precision_at_k(gold_scores, predicted_scores, k, threshold):
    order = sorted(range(len(gold_scores)), key=lambda i: predicted_scores[i], reverse=True)
    relevant_count = 0
    precision_sum = 0.0

    for rank, index in enumerate(order[:k], start=1):
        if gold_scores[index] >= threshold:
            relevant_count += 1
            precision_sum += relevant_count / rank

    total_relevant = sum(score >= threshold for score in gold_scores)

    if total_relevant == 0:
        return 0.0

    return precision_sum / min(total_relevant, k)


def precision_at_k(gold_scores, predicted_scores, k, threshold):
    order = sorted(range(len(gold_scores)), key=lambda i: predicted_scores[i], reverse=True)
    chosen = order[:k]

    if not chosen:
        return 0.0

    return sum(gold_scores[i] >= threshold for i in chosen) / len(chosen)


def pairwise_accuracy(gold_scores, predicted_scores):
    correct = 0
    total = 0

    for i in range(len(gold_scores)):
        for j in range(i + 1, len(gold_scores)):
            if gold_scores[i] == gold_scores[j]:
                continue

            total += 1
            gold_prefers_i = gold_scores[i] > gold_scores[j]
            pred_prefers_i = predicted_scores[i] > predicted_scores[j]

            if gold_prefers_i == pred_prefers_i:
                correct += 1

    return correct / total if total else 0.0


def summarize(target_results, k, acceptable_threshold):
    metrics = {
        f"ndcg@{k}": [],
        f"map@{k}": [],
        f"precision@{k}": [],
        "pairwise_accuracy": [],
    }

    for result in target_results:
        gold = result["gold_scores"]
        pred = result["predicted_scores"]
        metrics[f"ndcg@{k}"].append(ndcg_at_k(gold, pred, k))
        metrics[f"map@{k}"].append(
            average_precision_at_k(gold, pred, k, acceptable_threshold)
        )
        metrics[f"precision@{k}"].append(
            precision_at_k(gold, pred, k, acceptable_threshold)
        )
        metrics["pairwise_accuracy"].append(pairwise_accuracy(gold, pred))

    return {
        name: sum(values) / len(values) if values else 0.0
        for name, values in metrics.items()
    }


def candidate_score(candidate, config: RankerConfig):
    score = (
        config.lexical_weight * candidate["lexical"]
        + config.target_weight * candidate["target"]
        + config.semantic_weight * candidate["semantic"]
        + config.mlm_weight * candidate["mlm"]
        + config.substitute_weight * candidate["substitute"]
    )

    if candidate["lexical"] < 1.0:
        score *= config.lexical_precision_bias

    if candidate["semantic"] < config.semantic_threshold:
        score *= config.semantic_below_threshold_penalty

    if candidate["target"] < config.target_threshold:
        score *= config.target_below_threshold_penalty

    if candidate["mlm"] < config.mlm_threshold:
        score *= config.mlm_below_threshold_penalty

    if candidate["substitute"] < config.substitute_min_score:
        score *= config.substitute_below_min_penalty

    if candidate["substitute"] < config.substitute_low_score_threshold:
        score *= config.substitute_low_score_penalty

    return score


def normalize_scores(values):
    if not values:
        return []

    minimum = min(values)
    maximum = max(values)

    if maximum - minimum < 1e-9:
        return [0.5 for _ in values]

    return [(value - minimum) / (maximum - minimum) for value in values]


def rerank_stage_scores(base_scores, rerank_scores, config: RankerConfig):
    final_scores = list(normalize_scores(base_scores))
    pool_size = max(0, min(config.rerank_pool_size, len(base_scores)))

    if pool_size == 0:
        return base_scores

    pool_indexes = sorted(
        range(len(base_scores)),
        key=lambda index: base_scores[index],
        reverse=True,
    )[:pool_size]
    if not pool_indexes:
        return base_scores

    pool_base = normalize_scores([base_scores[index] for index in pool_indexes])
    pool_rerank = normalize_scores([rerank_scores[index] for index in pool_indexes])
    combined = []

    for base_score, rerank_score in zip(pool_base, pool_rerank):
        combined.append(
            1.0
            + (
                (1.0 - config.rerank_blend_alpha) * base_score
                + config.rerank_blend_alpha * rerank_score
            )
        )

    for index, score in zip(pool_indexes, combined):
        final_scores[index] = score

    return final_scores


def apply_config(feature_results, config: RankerConfig, learned_ranker=None):
    tuned = []

    for result in feature_results:
        if learned_ranker is None:
            base_scores = [
                candidate_score(candidate, config)
                for candidate in result["features"]
            ]
        else:
            base_scores = learned_ranker.predict_rows(result["features"])

        predicted_scores = rerank_stage_scores(
            base_scores,
            [candidate["rerank"] for candidate in result["features"]],
            config,
        )
        tuned.append(
            {
                **result,
                "predicted_scores": predicted_scores,
            }
        )

    return tuned


def tune_config(feature_results, k, acceptable_threshold):
    weight_presets = [
        {
            "lexical_weight": 0.25,
            "target_weight": 0.15,
            "semantic_weight": 0.15,
            "mlm_weight": 0.25,
            "substitute_weight": 0.0,
            "rerank_weight": 0.20,
        },
        {
            "lexical_weight": 0.30,
            "target_weight": 0.20,
            "semantic_weight": 0.10,
            "mlm_weight": 0.25,
            "substitute_weight": 0.0,
            "rerank_weight": 0.15,
        },
        {
            "lexical_weight": 0.20,
            "target_weight": 0.20,
            "semantic_weight": 0.10,
            "mlm_weight": 0.35,
            "substitute_weight": 0.0,
            "rerank_weight": 0.15,
        },
        {
            "lexical_weight": 0.20,
            "target_weight": 0.10,
            "semantic_weight": 0.15,
            "mlm_weight": 0.20,
            "substitute_weight": 0.30,
            "rerank_weight": 0.05,
        },
    ]
    semantic_thresholds = [0.60, 0.70, 0.75, 0.80]
    target_thresholds = [0.00, 0.10, 0.15, 0.25]
    mlm_thresholds = [0.00, 0.10, 0.16, 0.25]
    lexical_precision_biases = [1.0, 0.85, 0.70, 0.55]
    defaults = RankerConfig()
    semantic_penalties = [defaults.semantic_below_threshold_penalty]
    target_penalties = [defaults.target_below_threshold_penalty]
    mlm_penalties = [defaults.mlm_below_threshold_penalty]
    substitute_penalties = [defaults.substitute_below_min_penalty]
    substitute_min_scores = [0.00, 0.10, 0.15, 0.20, 0.25]
    substitute_low_score_thresholds = [0.15, 0.25, 0.35]
    substitute_low_score_penalties = [0.35, 0.50, 0.70]

    best_config = None
    best_metrics = None
    best_score = -1.0

    for weights in weight_presets:
        for semantic_threshold in semantic_thresholds:
            for target_threshold in target_thresholds:
                for mlm_threshold in mlm_thresholds:
                    for lexical_precision_bias in lexical_precision_biases:
                        for semantic_penalty in semantic_penalties:
                            for target_penalty in target_penalties:
                                for mlm_penalty in mlm_penalties:
                                    for substitute_penalty in substitute_penalties:
                                        for substitute_min_score in substitute_min_scores:
                                            for low_score_threshold in substitute_low_score_thresholds:
                                                for low_score_penalty in substitute_low_score_penalties:
                                                    config = RankerConfig(
                                                        **weights,
                                                        semantic_threshold=semantic_threshold,
                                                        target_threshold=target_threshold,
                                                        mlm_threshold=mlm_threshold,
                                                        lexical_precision_bias=lexical_precision_bias,
                                                        semantic_below_threshold_penalty=semantic_penalty,
                                                        target_below_threshold_penalty=target_penalty,
                                                        mlm_below_threshold_penalty=mlm_penalty,
                                                        substitute_below_min_penalty=substitute_penalty,
                                                        substitute_min_score=substitute_min_score,
                                                        substitute_low_score_threshold=low_score_threshold,
                                                        substitute_low_score_penalty=low_score_penalty,
                                                    )
                                                    results = apply_config(feature_results, config)
                                                    metrics = summarize(results, k, acceptable_threshold)
                                                    score = metrics[f"ndcg@{k}"]

                                                    if score > best_score:
                                                        best_score = score
                                                        best_config = config
                                                        best_metrics = metrics

    return best_config, best_metrics


def score_swords(pipeline, swords, max_targets=None):
    target_results = []

    for index, item in enumerate(iter_swords_targets(swords), start=1):
        if max_targets is not None and index > max_targets:
            break

        substitutes = [row["substitute"] for row in item["substitutes"]]
        gold_scores = [row["gold_score"] for row in item["substitutes"]]
        scored = pipeline.score_substitutes(
            item["context"],
            item["target"],
            substitutes,
            target_offset=item["offset"],
            apply_thresholds=False,
        )
        scored_by_word = {candidate.word.lower(): candidate for candidate in scored}
        features = []

        for substitute in substitutes:
            candidate = scored_by_word.get(substitute.lower())

            if candidate is None:
                features.append(
                    {
                        "lexical": 0.0,
                        "target": 0.0,
                        "semantic": 0.0,
                        "mlm": 0.0,
                        "substitute": 0.0,
                        "rerank": 0.0,
                    }
                )
                continue

            features.append(
                {
                    "lexical": candidate.lexical_score,
                    "target": candidate.target_score,
                    "semantic": candidate.semantic_score,
                    "mlm": candidate.mlm_score,
                    "substitute": candidate.substitute_score,
                    "rerank": candidate.rerank_score,
                }
            )

        target_results.append(
            {
                "target_id": item["target_id"],
                "target": item["target"],
                "gold_scores": gold_scores,
                "features": features,
            }
        )

        if index % 25 == 0:
            print(f"Scored {index} SWORDS targets")

    return target_results


def main():
    args = parse_args()
    swords, path = load_or_download_swords(
        split=args.split,
        data_dir=args.data_dir,
        path=args.swords_path,
    )
    print(f"Loaded SWORDS {args.split} from {path}")
    resource_swords_path, resource_swords_split = resolve_resource_swords(args, path)

    if resource_swords_path is None:
        print("SWORDS lexical-resource candidates disabled")
    else:
        print(
            "Using SWORDS lexical-resource candidates from "
            f"{resource_swords_split}: {resource_swords_path}"
        )

    pipeline = LexicalSubstitutionPipeline(
        ranker_config_path=args.ranker_config,
        substitute_classifier_model_name=args.substitute_model,
        substitute_positive_label="valid",
        substitute_threshold=args.substitute_threshold,
        swords_path=resource_swords_path,
    )
    feature_results = score_swords(pipeline, swords, max_targets=args.max_targets)
    learned_ranker = (
        load_learned_ranker(args.learned_ranker_model)
        if args.learned_ranker_model is not None
        else None
    )

    if args.tune:
        config, metrics = tune_config(
            feature_results,
            args.top_k,
            args.acceptable_threshold,
        )
        Path(args.output_config).write_text(
            json.dumps(config.__dict__, indent=2),
            encoding="utf-8",
        )
        print(f"Saved tuned ranker config to {args.output_config}")
    else:
        config = pipeline.ranker_config
        metrics = summarize(
            apply_config(feature_results, config, learned_ranker=learned_ranker),
            args.top_k,
            args.acceptable_threshold,
        )

    payload = {
        "split": args.split,
        "resource_swords_split": resource_swords_split,
        "resource_swords_path": resource_swords_path,
        "targets": len(feature_results),
        "top_k": args.top_k,
        "acceptable_threshold": args.acceptable_threshold,
        "ranker_config": config.__dict__,
        "learned_ranker_model": args.learned_ranker_model,
        "metrics": metrics,
    }
    print(json.dumps(payload, indent=2))

    if args.output_metrics is not None:
        Path(args.output_metrics).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
