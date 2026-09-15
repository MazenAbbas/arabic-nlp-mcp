"""Command-line interface for scripts and data pipelines."""

from __future__ import annotations

import argparse
import json
import sys

from .diacritize import diacritize
from .dialect import detect_dialect
from .models import ResultModel
from .normalize import normalize
from .search import prepare_batch, prepare_for_search
from .sentiment import analyze_sentiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arabic-nlp", description="Arabic NLP utilities")
    parser.add_argument("--pretty", action="store_true", help="indent JSON output")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("normalize", "dialect", "diacritize"):
        command = sub.add_parser(name)
        command.add_argument("text", nargs="?", help="text; reads stdin when omitted")
    search = sub.add_parser("search")
    search.add_argument("text", nargs="?", help="text; reads stdin when omitted")
    search.add_argument(
        "--profile", choices=("conservative", "search", "aggressive"), default="search"
    )
    search.add_argument("--jsonl", action="store_true", help="process one text per input line")
    sentiment = sub.add_parser("sentiment")
    sentiment.add_argument("text", nargs="?", help="text; reads stdin when omitted")
    sentiment.add_argument("--backend", choices=("auto", "transformer", "lexicon"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    # JSON is UTF-8 on every platform, including redirected Windows consoles.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    if args.command == "search" and args.jsonl:
        if args.text is not None:
            raise SystemExit("search --jsonl reads stdin; do not pass positional text")
        lines = (line.rstrip("\r\n") for line in sys.stdin)
        for batch_result in prepare_batch(lines, profile=args.profile):
            print(json.dumps(batch_result.model_dump(), ensure_ascii=False))
        return 0
    text = args.text if args.text is not None else sys.stdin.read()
    result: ResultModel
    if args.command == "normalize":
        result = normalize(text)
    elif args.command == "search":
        result = prepare_for_search(text, profile=args.profile)
    elif args.command == "dialect":
        result = detect_dialect(text)
    elif args.command == "sentiment":
        result = analyze_sentiment(text, backend=args.backend)
    else:
        result = diacritize(text)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
