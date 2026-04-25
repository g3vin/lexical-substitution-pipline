from __future__ import annotations

from collections import defaultdict
import gzip
import json
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve


SWORDS_URLS = {
    "dev": [
        "https://raw.githubusercontent.com/p-lambda/swords/main/assets/parsed/swords-v1.1_dev.json.gz",
    ],
    "test": [
        "https://raw.githubusercontent.com/p-lambda/swords/main/assets/parsed/swords-v1.1_test.json.gz",
    ],
}


def default_swords_path(split: str, data_dir: str | Path = "data/swords") -> Path:
    return Path(data_dir) / f"swords-v1.1_{split}.json.gz"


def download_swords(split: str, data_dir: str | Path = "data/swords") -> Path:
    if split not in SWORDS_URLS:
        raise ValueError(f"Unknown SWORDS split: {split}")

    output_path = default_swords_path(split, data_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        return output_path

    errors = []

    for url in SWORDS_URLS[split]:
        try:
            urlretrieve(url, output_path)
            return output_path
        except Exception as error:
            errors.append(f"{url}: {error}")

    raise RuntimeError(
        "Could not download SWORDS. Download the split manually from "
        "https://github.com/p-lambda/swords and place it at "
        f"{output_path}.\n" + "\n".join(errors)
    )


def load_swords(path: str | Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_or_download_swords(
    split: str = "dev",
    data_dir: str | Path = "data/swords",
    path: str | Path | None = None,
):
    dataset_path = Path(path) if path is not None else download_swords(split, data_dir)
    return load_swords(dataset_path), dataset_path


def target_to_substitute_ids(swords) -> dict[str, list[str]]:
    tid_to_sids = defaultdict(list)

    for sid, substitute in swords["substitutes"].items():
        tid_to_sids[substitute["target_id"]].append(sid)

    return tid_to_sids


def label_score(labels: list[str]) -> float:
    if not labels:
        return 0.0

    return labels.count("TRUE") / len(labels)


def iter_swords_targets(swords) -> Iterable[dict]:
    tid_to_sids = target_to_substitute_ids(swords)

    for target_id, target in swords["targets"].items():
        context = swords["contexts"][target["context_id"]]["context"]
        substitute_rows = []

        for substitute_id in tid_to_sids[target_id]:
            substitute = swords["substitutes"][substitute_id]["substitute"]
            labels = swords["substitute_labels"][substitute_id]
            substitute_rows.append(
                {
                    "substitute_id": substitute_id,
                    "substitute": substitute,
                    "gold_score": label_score(labels),
                    "labels": labels,
                }
            )

        yield {
            "target_id": target_id,
            "context": context,
            "target": target["target"],
            "offset": target["offset"],
            "pos": target.get("pos"),
            "substitutes": substitute_rows,
        }
