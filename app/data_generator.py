
#Synthetic Cypress log generator.

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _gauss_clamp(mean: float, std: float, lo: float, hi: float, rng: random.Random) -> float:
    return max(lo, min(hi, rng.gauss(mean, std)))


def _generate_timeout(rng: random.Random) -> dict:
    return {
        "execution_time_ms": _gauss_clamp(15_000, 3_000, 8_000, 30_000, rng),
        "failed_step_index": int(_gauss_clamp(8, 3, 1, 20, rng)),
        "retry_count": int(_gauss_clamp(3, 1, 1, 5, rng)),
        "error_code": None,
        "dom_selector_depth": int(_gauss_clamp(7, 2, 3, 15, rng)),
        "network_call_count": int(_gauss_clamp(5, 2, 1, 15, rng)),
        "label": "timeout",
    }


def _generate_network_error(rng: random.Random) -> dict:
    code = rng.choice([0, 400, 401, 403, 404, 500, 502, 503, 504])
    return {
        "execution_time_ms": _gauss_clamp(2_000, 800, 500, 6_000, rng),
        "failed_step_index": int(_gauss_clamp(3, 2, 1, 10, rng)),
        "retry_count": int(_gauss_clamp(1, 1, 0, 3, rng)),
        "error_code": code,
        "dom_selector_depth": int(_gauss_clamp(3, 1, 1, 8, rng)),
        "network_call_count": int(_gauss_clamp(12, 4, 3, 25, rng)),
        "label": "network_error",
    }


def _generate_ui_bug(rng: random.Random) -> dict:
    return {
        "execution_time_ms": _gauss_clamp(4_000, 1_200, 1_500, 9_000, rng),
        "failed_step_index": int(_gauss_clamp(5, 2, 1, 15, rng)),
        "retry_count": int(_gauss_clamp(0, 0.5, 0, 2, rng)),
        "error_code": None,
        "dom_selector_depth": int(_gauss_clamp(4, 1, 1, 10, rng)),
        "network_call_count": int(_gauss_clamp(4, 2, 0, 12, rng)),
        "label": "ui_bug",
    }


_GENERATORS = {
    "timeout": _generate_timeout,
    "network_error": _generate_network_error,
    "ui_bug": _generate_ui_bug,
}


def generate_dataset(n: int = 600, seed: int = 42) -> list[dict]:
    """Return *n* synthetic log entries (balanced across 3 classes)."""
    rng = random.Random(seed)  # noqa: S311
    labels = list(_GENERATORS.keys())
    records: list[dict] = []
    per_class = n // len(labels)
    for label in labels:
        gen = _GENERATORS[label]
        records.extend(gen(rng) for _ in range(per_class))
    for label in labels[: n % len(labels)]:
        records.append(_GENERATORS[label](rng))
    rng.shuffle(records)
    return records


def save_jsonl(records: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Cypress log dataset")
    parser.add_argument("--n", type=int, default=600, help="Total number of records")
    parser.add_argument("--out", type=str, default="data/logs.jsonl", help="Output .jsonl path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    records = generate_dataset(args.n, args.seed)
    save_jsonl(records, args.out)
    print(f"Generated {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
