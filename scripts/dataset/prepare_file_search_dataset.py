"""Create unambiguous UTF-8 text shards for Gemini File Search ingestion."""

import argparse
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "amazon_final_intent_dataset.csv"
DEFAULT_OUTPUT = ROOT / "file_search_dataset"


def stable_case_id(customer_text: str, amazon_reply: str) -> str:
    digest = hashlib.sha256(f"{customer_text}\0{amazon_reply}".encode("utf-8")).hexdigest()[:16]
    return f"TWCS-AMAZON-{digest}"


def language_hint(text: str) -> str:
    if any("\u0900" <= char <= "\u097f" for char in text):
        return "hi"
    if any("\u3040" <= char <= "\u30ff" for char in text):
        return "ja"
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return "zh"
    return "unknown"


def prepare(input_path: Path, output_dir: Path, shard_size: int = 750) -> tuple[int, list[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    handle = None
    count = 0
    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                if count % shard_size == 0:
                    if handle:
                        handle.close()
                    path = output_dir / f"amazon_support_cases_{count // shard_size + 1:03d}.txt"
                    files.append(path)
                    handle = path.open("w", encoding="utf-8", newline="\n")
                customer = (row.get("customer_text") or "").strip()
                reply = (row.get("amazon_reply") or "").strip()
                case_id = stable_case_id(customer, reply)
                handle.write(
                    f"CASE_ID: {case_id}\nINTENT: {(row.get('intent') or 'UNKNOWN').strip()}\n"
                    f"LANGUAGE: {language_hint(customer)}\nCUSTOMER_ISSUE:\n{customer}\n\n"
                    f"HISTORICAL_RESOLUTION:\n{reply}\n\nSOURCE: TWCS AmazonHelp\n"
                    "BRAND: AmazonHelp\n--- END CASE ---\n\n"
                )
                count += 1
    finally:
        if handle:
            handle.close()
    return count, files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shard-size", type=int, default=750)
    args = parser.parse_args()
    total, shards = prepare(args.input, args.output, args.shard_size)
    print(f"Prepared {total:,} cases in {len(shards)} UTF-8 text shards at {args.output}")
