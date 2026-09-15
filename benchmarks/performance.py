"""Reproducible local throughput check; not an accuracy benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

from arabic_nlp_mcp.search import prepare_for_search

SAMPLES = (
    "إِنَّ معالجةَ النصوص العربية مهمة للبحث والاسترجاع.",
    "رقم الطلب ١٢٣٤٥ وحالته قيد المعالجة — GPT-5",
    "هــذا نــص يحتوي محارف خفية\u200f وتشكيلًا وأرقامًا ۲۰۲۶.",
)


def measure(iterations: int) -> dict[str, object]:
    for text in SAMPLES:
        prepare_for_search(text)
    runs = []
    for _ in range(5):
        started = time.perf_counter()
        for index in range(iterations):
            prepare_for_search(SAMPLES[index % len(SAMPLES)])
        runs.append(time.perf_counter() - started)
    median = statistics.median(runs)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "iterations_per_run": iterations,
        "runs": len(runs),
        "median_seconds": round(median, 6),
        "median_items_per_second": round(iterations / median, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10_000)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    print(json.dumps(measure(args.iterations), indent=2))


if __name__ == "__main__":
    main()
