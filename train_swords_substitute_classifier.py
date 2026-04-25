import argparse
import random
from dataclasses import dataclass

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from swords_utils import iter_swords_targets, load_or_download_swords


TARGET_START = "<target>"
TARGET_END = "</target>"


@dataclass
class TrainingConfig:
    model_name: str = "roberta-base"
    output_dir: str = "./substitute-roberta"
    split: str = "dev"
    data_dir: str = "data/swords"
    swords_path: str | None = None
    max_length: int = 192
    learning_rate: float = 2e-5
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    logging_steps: int = 50
    validation_fraction: float = 0.15
    positive_threshold: float = 0.5
    max_examples: int | None = None
    seed: int = 13


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="Train a substitute-quality regression model from SWORDS soft labels."
    )
    parser.add_argument("--model-name", default=TrainingConfig.model_name)
    parser.add_argument("--output-dir", default=TrainingConfig.output_dir)
    parser.add_argument("--split", default=TrainingConfig.split, choices=["dev", "test"])
    parser.add_argument("--data-dir", default=TrainingConfig.data_dir)
    parser.add_argument("--swords-path", default=None)
    parser.add_argument("--max-length", type=int, default=TrainingConfig.max_length)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--train-batch-size", type=int, default=TrainingConfig.train_batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=TrainingConfig.eval_batch_size)
    parser.add_argument("--num-train-epochs", type=int, default=TrainingConfig.num_train_epochs)
    parser.add_argument("--weight-decay", type=float, default=TrainingConfig.weight_decay)
    parser.add_argument("--logging-steps", type=int, default=TrainingConfig.logging_steps)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=TrainingConfig.validation_fraction,
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=TrainingConfig.positive_threshold,
        help="Only used for reporting threshold accuracy; training uses gold scores.",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    args = parser.parse_args()
    return TrainingConfig(**vars(args))


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def mark_target(sentence: str, offset: int, target: str) -> str:
    end = offset + len(target)
    return (
        sentence[:offset]
        + TARGET_START
        + " "
        + sentence[offset:end]
        + " "
        + TARGET_END
        + sentence[end:]
    )


def replace_and_mark(sentence: str, offset: int, target: str, substitute: str) -> str:
    end = offset + len(target)
    substituted = sentence[:offset] + substitute + sentence[end:]
    return mark_target(substituted, offset, substitute)


def build_rows(swords, positive_threshold: float):
    rows = []

    for item in iter_swords_targets(swords):
        original = mark_target(item["context"], item["offset"], item["target"])

        for substitute in item["substitutes"]:
            rows.append(
                {
                    "original": original,
                    "substitution": replace_and_mark(
                        item["context"],
                        item["offset"],
                        item["target"],
                        substitute["substitute"],
                    ),
                    "labels": float(substitute["gold_score"]),
                    "gold_score": substitute["gold_score"],
                }
            )

    return rows


def split_rows(rows, validation_fraction: float, seed: int):
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    validation_size = max(1, int(len(rows) * validation_fraction))
    return rows[validation_size:], rows[:validation_size]


def tokenize_rows(rows, tokenizer, max_length: int):
    dataset = Dataset.from_list(rows)

    def tokenize(batch):
        return tokenizer(
            batch["original"],
            batch["substitution"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns(["original", "substitution", "gold_score"])
    tokenized.set_format(type="torch")
    return tokenized


def move_batch_to_device(batch, device: torch.device):
    return {key: value.to(device) for key, value in batch.items()}


def evaluate_model(
    model,
    dataloader,
    device: torch.device,
    positive_threshold: float,
):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0
    total_absolute_error = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            labels = batch["labels"]
            scores = outputs.logits[:, 0].clamp(0.0, 1.0)
            predictions = scores >= positive_threshold
            binary_labels = labels >= positive_threshold
            batch_size = labels.size(0)
            total_loss += outputs.loss.item() * batch_size
            total_absolute_error += torch.abs(scores - labels).sum().item()
            total_correct += (predictions == binary_labels).sum().item()
            total_examples += batch_size

    return {
        "mse": total_loss / total_examples,
        "rmse": (total_loss / total_examples) ** 0.5,
        "mae": total_absolute_error / total_examples,
        "threshold_accuracy": total_correct / total_examples,
    }


def main():
    config = parse_args()
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    swords, path = load_or_download_swords(
        split=config.split,
        data_dir=config.data_dir,
        path=config.swords_path,
    )
    rows = build_rows(swords, config.positive_threshold)

    if config.max_examples is not None:
        rows = rows[: config.max_examples]

    train_rows, validation_rows = split_rows(
        rows,
        config.validation_fraction,
        config.seed,
    )
    print(
        f"Loaded {len(rows)} soft-labeled SWORDS candidates from {path}; "
        f"{len(train_rows)} train / {len(validation_rows)} validation"
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [TARGET_START, TARGET_END]}
    )
    train_dataset = tokenize_rows(train_rows, tokenizer, config.max_length)
    validation_dataset = tokenize_rows(validation_rows, tokenizer, config.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=1,
        problem_type="regression",
    )
    model.resize_token_embeddings(len(tokenizer))

    device = get_device()
    model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.eval_batch_size,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * config.num_train_epochs,
    )

    best_mse = float("inf")
    global_step = 0

    for epoch in range(1, config.num_train_epochs + 1):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            global_step += 1

            if global_step % config.logging_steps == 0:
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"train_loss={running_loss / step:.4f}"
                )

        metrics = evaluate_model(
            model,
            validation_loader,
            device,
            config.positive_threshold,
        )
        print(
            f"epoch={epoch} validation_mse={metrics['mse']:.4f} "
            f"validation_rmse={metrics['rmse']:.4f} "
            f"validation_mae={metrics['mae']:.4f} "
            f"threshold_accuracy={metrics['threshold_accuracy']:.4f}"
        )

        if metrics["mse"] < best_mse:
            best_mse = metrics["mse"]
            model.save_pretrained(config.output_dir)
            tokenizer.save_pretrained(config.output_dir)
            print(
                f"Saved new best model to {config.output_dir} "
                f"(mse={best_mse:.4f})"
            )

    print(f"Finished training. Best validation MSE: {best_mse:.4f}")


if __name__ == "__main__":
    main()
