import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lexsub-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from evaluate_swords import apply_config, score_swords, summarize
from pipeline.learned_ranker import load_learned_ranker
from pipeline import LexicalSubstitutionPipeline, RankerConfig
from swords_utils import iter_swords_targets, load_or_download_swords


FEATURE_COLUMNS = ["lexical", "target", "semantic", "mlm", "substitute", "rerank"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate report plots for SWORDS lexical substitution experiments."
    )
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--data-dir", default="data/swords")
    parser.add_argument("--swords-path", default=None)
    parser.add_argument("--output-dir", default="report_plots")
    parser.add_argument("--ranker-config", default=None)
    parser.add_argument("--learned-ranker-model", default=None)
    parser.add_argument("--substitute-model", default=None)
    parser.add_argument("--substitute-threshold", type=float, default=0.5)
    parser.add_argument("--acceptable-threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument(
        "--include-model-plots",
        action="store_true",
        help="Run the pipeline over SWORDS candidates and plot model behavior.",
    )
    return parser.parse_args()


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_ranker_config(path: str | None) -> RankerConfig:
    if path is None:
        return RankerConfig()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid_keys = set(RankerConfig.__dataclass_fields__)
    return RankerConfig(**{key: value for key, value in data.items() if key in valid_keys})


def swords_dataframe(swords) -> pd.DataFrame:
    rows = []

    for item in iter_swords_targets(swords):
        for substitute in item["substitutes"]:
            rows.append(
                {
                    "target_id": item["target_id"],
                    "target": item["target"],
                    "pos": item["pos"],
                    "substitute": substitute["substitute"],
                    "gold_score": substitute["gold_score"],
                    "label": int(substitute["gold_score"] >= 0.5),
                    "is_single_word": int(" " not in substitute["substitute"].strip()),
                    "substitute_length": len(substitute["substitute"]),
                }
            )

    return pd.DataFrame(rows)


def save_current_figure(output_dir: Path, filename: str):
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=220)
    plt.close()


def plot_dataset_overview(df: pd.DataFrame, output_dir: Path):
    plt.figure(figsize=(7, 4))
    sns.histplot(df["gold_score"], bins=11)
    plt.title("SWORDS Gold Substitute Score Distribution")
    plt.xlabel("Gold score: fraction of TRUE labels")
    plt.ylabel("Candidate count")
    save_current_figure(output_dir, "gold_score_distribution.png")

    plt.figure(figsize=(7, 4))
    pos_counts = df.drop_duplicates("target_id")["pos"].fillna("UNKNOWN").value_counts()
    sns.barplot(x=pos_counts.index, y=pos_counts.values)
    plt.title("Target POS Distribution")
    plt.xlabel("POS")
    plt.ylabel("Target count")
    save_current_figure(output_dir, "target_pos_distribution.png")

    plt.figure(figsize=(7, 4))
    counts = df.groupby("target_id").size()
    sns.histplot(counts, bins=25)
    plt.title("Number of Candidate Substitutes per Target")
    plt.xlabel("Candidates per target")
    plt.ylabel("Target count")
    save_current_figure(output_dir, "candidates_per_target.png")

    plt.figure(figsize=(6, 4))
    wordform_counts = df["is_single_word"].map({1: "single word", 0: "multi-word"}).value_counts()
    sns.barplot(x=wordform_counts.index, y=wordform_counts.values)
    plt.title("Single-Word vs Multi-Word Gold Candidates")
    plt.xlabel("")
    plt.ylabel("Candidate count")
    save_current_figure(output_dir, "single_vs_multi_word_candidates.png")


def target_pos_lookup(df: pd.DataFrame) -> dict[str, str]:
    return (
        df.drop_duplicates("target_id")
        .assign(pos=lambda frame: frame["pos"].fillna("UNKNOWN"))
        .set_index("target_id")["pos"]
        .to_dict()
    )


def flatten_feature_results(
    feature_results,
    tuned_results,
    target_pos_by_id: dict[str, str] | None = None,
) -> pd.DataFrame:
    rows = []
    target_pos_by_id = target_pos_by_id or {}

    for feature_result, tuned_result in zip(feature_results, tuned_results):
        for index, feature in enumerate(feature_result["features"]):
            rows.append(
                {
                    "target_id": feature_result["target_id"],
                    "target": feature_result["target"],
                    "pos": target_pos_by_id.get(feature_result["target_id"], "UNKNOWN"),
                    "gold_score": feature_result["gold_scores"][index],
                    "label": int(feature_result["gold_scores"][index] >= 0.5),
                    "final_score": tuned_result["predicted_scores"][index],
                    **feature,
                }
            )

    return pd.DataFrame(rows)


def ablation_config(config: RankerConfig, feature_name: str) -> RankerConfig:
    if feature_name == "lexical":
        return replace(config, lexical_weight=0.0)
    if feature_name == "target":
        return replace(config, target_weight=0.0, target_threshold=0.0)
    if feature_name == "semantic":
        return replace(config, semantic_weight=0.0, semantic_threshold=0.0)
    if feature_name == "mlm":
        return replace(config, mlm_weight=0.0, mlm_threshold=0.0)
    if feature_name == "substitute":
        return replace(
            config,
            substitute_weight=0.0,
            substitute_min_score=0.0,
            substitute_low_score_threshold=0.0,
            substitute_low_score_penalty=1.0,
        )
    if feature_name == "rerank":
        return replace(config, rerank_weight=0.0)
    raise ValueError(f"Unknown feature for ablation: {feature_name}")


def feature_ablation_metrics(
    feature_results,
    config: RankerConfig,
    acceptable_threshold: float,
    top_k: int,
    learned_ranker=None,
) -> pd.DataFrame:
    rows = []
    variants = ["All features"] + [f"No {feature}" for feature in FEATURE_COLUMNS]

    for variant in variants:
        variant_config = config
        variant_results = feature_results

        if variant != "All features":
            feature_name = variant.removeprefix("No ")
            if learned_ranker is None:
                variant_config = ablation_config(config, feature_name)
            else:
                variant_results = ablate_feature_results(feature_results, feature_name)
                if feature_name == "rerank":
                    variant_config = replace(
                        config,
                        rerank_pool_size=0,
                        rerank_blend_alpha=0.0,
                    )

        metrics = summarize(
            apply_config(
                variant_results,
                variant_config,
                learned_ranker=learned_ranker,
            ),
            top_k,
            acceptable_threshold,
        )
        rows.append(
            {
                "variant": variant,
                "ndcg": metrics[f"ndcg@{top_k}"],
                "map": metrics[f"map@{top_k}"],
                "precision": metrics[f"precision@{top_k}"],
                "pairwise_accuracy": metrics["pairwise_accuracy"],
            }
        )

    return pd.DataFrame(rows)


def ablate_feature_results(feature_results, feature_name: str):
    ablated = []

    for result in feature_results:
        ablated_features = []
        for feature_row in result["features"]:
            updated_row = dict(feature_row)
            if feature_name in updated_row:
                updated_row[feature_name] = 0.0
            ablated_features.append(updated_row)

        ablated.append(
            {
                **result,
                "features": ablated_features,
            }
        )

    return ablated


def metrics_by_pos(
    tuned_results,
    target_pos_by_id: dict[str, str],
    acceptable_threshold: float,
    top_k: int,
) -> pd.DataFrame:
    grouped_results: dict[str, list[dict]] = {}

    for result in tuned_results:
        pos = target_pos_by_id.get(result["target_id"], "UNKNOWN")
        grouped_results.setdefault(pos, []).append(result)

    rows = []

    for pos, pos_results in grouped_results.items():
        metrics = summarize(pos_results, top_k, acceptable_threshold)
        rows.append(
            {
                "pos": pos,
                "targets": len(pos_results),
                "ndcg": metrics[f"ndcg@{top_k}"],
                "map": metrics[f"map@{top_k}"],
                "precision": metrics[f"precision@{top_k}"],
                "pairwise_accuracy": metrics["pairwise_accuracy"],
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["targets", "pos"],
        ascending=[False, True],
    )


def plot_model_overview(
    model_df: pd.DataFrame,
    metrics_by_k: pd.DataFrame,
    ablation_df: pd.DataFrame,
    pos_metrics_df: pd.DataFrame,
    output_dir: Path,
    top_k: int,
):
    finite_df = model_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["final_score"])
    finite_df = finite_df[finite_df["final_score"] > -1e8]

    if finite_df.empty:
        return

    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=finite_df,
        x="gold_score",
        y="final_score",
        hue="label",
        alpha=0.6,
    )
    plt.title("Final Model Score vs SWORDS Gold Score")
    plt.xlabel("SWORDS gold score")
    plt.ylabel("Pipeline final score")
    save_current_figure(output_dir, "final_score_vs_gold.png")

    plt.figure(figsize=(9, 5))
    melted = finite_df.melt(
        id_vars=["label"],
        value_vars=FEATURE_COLUMNS + ["final_score"],
        var_name="feature",
        value_name="score",
    )
    sns.boxplot(data=melted, x="feature", y="score", hue="label")
    plt.title("Feature Score Distributions by Gold Validity")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.xticks(rotation=30, ha="right")
    save_current_figure(output_dir, "feature_distributions_by_label.png")

    plt.figure(figsize=(8, 6))
    corr = finite_df[FEATURE_COLUMNS + ["final_score", "gold_score"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Feature Correlation Heatmap")
    save_current_figure(output_dir, "feature_correlation_heatmap.png")

    plt.figure(figsize=(7, 4))
    sns.lineplot(data=metrics_by_k, x="k", y="ndcg", marker="o", label="NDCG")
    sns.lineplot(data=metrics_by_k, x="k", y="map", marker="o", label="MAP")
    sns.lineplot(data=metrics_by_k, x="k", y="precision", marker="o", label="Precision")
    plt.title("Ranking Metrics by K")
    plt.xlabel("K")
    plt.ylabel("Metric")
    plt.ylim(0, 1)
    save_current_figure(output_dir, "ranking_metrics_by_k.png")

    ablation_plot_df = ablation_df.melt(
        id_vars=["variant"],
        value_vars=["ndcg", "map", "precision"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=ablation_plot_df, x="variant", y="score", hue="metric")
    plt.title(f"Feature Ablation Metrics at K={top_k}")
    plt.xlabel("")
    plt.ylabel("Metric")
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    save_current_figure(output_dir, "feature_ablation_at_k.png")

    pos_plot_df = pos_metrics_df.copy()
    pos_plot_df["pos_label"] = pos_plot_df.apply(
        lambda row: f'{row["pos"]} (n={row["targets"]})',
        axis=1,
    )
    pos_plot_df = pos_plot_df.melt(
        id_vars=["pos_label"],
        value_vars=["ndcg", "map", "precision"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=pos_plot_df, x="pos_label", y="score", hue="metric")
    plt.title(f"Ranking Quality by Target POS at K={top_k}")
    plt.xlabel("")
    plt.ylabel("Metric")
    plt.ylim(0, 1)
    save_current_figure(output_dir, "ranking_metrics_by_pos.png")

    if finite_df["label"].nunique() == 2:
        y_true = finite_df["label"].to_numpy()
        y_score = finite_df["final_score"].to_numpy()

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title("Final Score ROC Curve")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.legend()
        save_current_figure(output_dir, "final_score_roc.png")

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        plt.figure(figsize=(5, 5))
        plt.plot(recall, precision, label=f"AP={ap:.3f}")
        plt.title("Final Score Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        save_current_figure(output_dir, "final_score_precision_recall.png")

        y_pred = (finite_df["final_score"] >= finite_df["final_score"].median()).astype(int)
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
        plt.title("Median-Threshold Confusion Matrix")
        save_current_figure(output_dir, "final_score_confusion_matrix.png")


def metrics_for_k(
    feature_results,
    config: RankerConfig,
    acceptable_threshold: float,
    max_k: int,
    learned_ranker=None,
):
    from evaluate_swords import apply_config, summarize

    rows = []
    tuned = apply_config(feature_results, config, learned_ranker=learned_ranker)

    for k in range(1, max_k + 1):
        metrics = summarize(tuned, k, acceptable_threshold)
        rows.append(
            {
                "k": k,
                "ndcg": metrics[f"ndcg@{k}"],
                "map": metrics[f"map@{k}"],
                "precision": metrics[f"precision@{k}"],
                "pairwise_accuracy": metrics["pairwise_accuracy"],
            }
        )

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    swords, path = load_or_download_swords(
        split=args.split,
        data_dir=args.data_dir,
        path=args.swords_path,
    )
    print(f"Loaded SWORDS {args.split} from {path}")

    df = swords_dataframe(swords)
    df.to_csv(output_dir / "swords_candidates.csv", index=False)
    plot_dataset_overview(df, output_dir)
    target_pos_by_id = target_pos_lookup(df)

    summary = {
        "split": args.split,
        "swords_path": str(path),
        "targets": int(df["target_id"].nunique()),
        "candidates": int(len(df)),
        "single_word_candidate_rate": float(df["is_single_word"].mean()),
        "mean_gold_score": float(df["gold_score"].mean()),
    }

    if args.include_model_plots:
        config = load_ranker_config(args.ranker_config)
        learned_ranker = (
            load_learned_ranker(args.learned_ranker_model)
            if args.learned_ranker_model is not None
            else None
        )
        pipeline = LexicalSubstitutionPipeline(
            ranker_config=config,
            substitute_classifier_model_name=args.substitute_model,
            substitute_positive_label="valid",
            substitute_threshold=args.substitute_threshold,
        )
        feature_results = score_swords(
            pipeline,
            swords,
            max_targets=args.max_targets,
        )
        tuned_results = apply_config(feature_results, config, learned_ranker=learned_ranker)
        model_df = flatten_feature_results(feature_results, tuned_results, target_pos_by_id)
        metrics_by_k = metrics_for_k(
            feature_results,
            config,
            args.acceptable_threshold,
            args.top_k,
            learned_ranker=learned_ranker,
        )
        ablation_df = feature_ablation_metrics(
            feature_results,
            config,
            args.acceptable_threshold,
            args.top_k,
            learned_ranker=learned_ranker,
        )
        pos_metrics_df = metrics_by_pos(
            tuned_results,
            target_pos_by_id,
            args.acceptable_threshold,
            args.top_k,
        )
        model_df.to_csv(output_dir / "model_candidate_features.csv", index=False)
        metrics_by_k.to_csv(output_dir / "ranking_metrics_by_k.csv", index=False)
        ablation_df.to_csv(output_dir / "feature_ablation_at_k.csv", index=False)
        pos_metrics_df.to_csv(output_dir / "ranking_metrics_by_pos.csv", index=False)
        plot_model_overview(
            model_df,
            metrics_by_k,
            ablation_df,
            pos_metrics_df,
            output_dir,
            args.top_k,
        )
        summary["model_targets"] = len(feature_results)
        summary["learned_ranker_model"] = args.learned_ranker_model
        summary["model_metrics_at_top_k"] = summarize(
            tuned_results,
            args.top_k,
            args.acceptable_threshold,
        )

    (output_dir / "plot_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"Saved plots and summaries to {output_dir}")


if __name__ == "__main__":
    main()
