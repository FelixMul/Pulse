"""
Precompute predictions for the email dataset using local Ollama models.

- Outputs (separate per model):
  - data/predictions_small.csv  (llama3.1:8b-instruct)
  - data/predictions_big.csv    (gpt-oss:20b)
- Resume behavior: skips rows whose recipe_id already exists in the target CSV
  unless --overwrite is provided.

Usage:
  uv run python evaluate_emails.py --model small [--overwrite] [--limit N] [--verbose]
  uv run python evaluate_emails.py --model big   [--overwrite] [--limit N] [--verbose]
"""

import argparse
import asyncio
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

# Import backend modules
sys.path.insert(0, str(Path(__file__).parent / "backend"))
from app.llm_processor import LLMProcessor  # type: ignore


DATASET_PATH = Path("synthetic_data_creation/data_creation/emails_data/email_dataset_final.csv")
OUTPUT_DIR = Path("data")

MODEL_MAP = {
    "small": "llama3.1:8b-instruct",
    "big": "gpt-oss:20b",
}


def run_coro(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)


def load_existing_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("recipe_id")
            if rid:
                rows[rid] = row
    return rows


def save_predictions_header_if_needed(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "recipe_id",
                "model_tag",
                "pred_topic",
                "pred_sentiment",
                "confidence",
                "summary",
                "timestamp",
            ],
        )
        writer.writeheader()


def append_prediction(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "recipe_id",
                "model_tag",
                "pred_topic",
                "pred_sentiment",
                "confidence",
                "summary",
                "timestamp",
            ],
        )
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute LLM predictions for emails")
    parser.add_argument("--model", choices=["small", "big"], required=True, help="Model size to use")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing predictions file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of emails to process")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging output")
    args = parser.parse_args()

    model_key = args.model
    model_tag = MODEL_MAP[model_key]
    out_path = OUTPUT_DIR / f"predictions_{model_key}.csv"

    if args.overwrite and out_path.exists():
        out_path.unlink()

    save_predictions_header_if_needed(out_path)
    existing = load_existing_predictions(out_path)

    # Load dataset
    if not DATASET_PATH.exists():
        print(f"Dataset not found: {DATASET_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)
    total = len(df)
    if args.limit is not None:
        df = df.head(args.limit)

    processor = LLMProcessor()

    # Check model availability
    ok = run_coro(processor.test_connection(model_tag=model_tag))
    if not ok:
        print(f"Model '{model_tag}' not available in Ollama. Make sure it's pulled and running.")
        sys.exit(1)

    processed = 0
    failed = 0
    start_time = time.time()
    if args.verbose:
        print(f"Starting evaluation with model '{model_tag}' on {len(df)} emails (of {total})")

    for idx, row in df.iterrows():
        recipe_id = str(row.get("recipe_id", "")).strip()
        email_text = str(row.get("email_body", ""))

        if not recipe_id or not email_text:
            continue

        if (not args.overwrite) and (recipe_id in existing):
            continue

        try:
            result = run_coro(processor.analyze_email(email_text, model_tag=model_tag))
            status = result.get("status", "success")
            pred_row = {
                "recipe_id": recipe_id,
                "model_tag": model_tag,
                "pred_topic": result.get("topic", "other"),
                "pred_sentiment": result.get("sentiment", "neutral"),
                "confidence": result.get("confidence", 0.0),
                "summary": result.get("summary", ""),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if status != "success":
                failed += 1
                if args.verbose:
                    print(f"[WARN] {recipe_id}: analysis status={status}")
        except Exception as e:
            failed += 1
            pred_row = {
                "recipe_id": recipe_id,
                "model_tag": model_tag,
                "pred_topic": "other",
                "pred_sentiment": "neutral",
                "confidence": 0.1,
                "summary": f"Error: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        append_prediction(out_path, pred_row)
        processed += 1

        # progress display
        if args.verbose:
            if processed % 25 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {processed}/{len(df)} | failed {failed} | {elapsed/60:.1f} min")
        else:
            # simple in-place progress bar
            width = 40
            ratio = processed / max(len(df), 1)
            filled = int(width * ratio)
            bar = "#" * filled + "-" * (width - filled)
            print(f"\r[{bar}] {processed}/{len(df)} | failed {failed}", end="", flush=True)

    elapsed = time.time() - start_time
    if not args.verbose:
        # ensure newline after in-place bar
        print()
    print(f"Done. Processed {processed}/{len(df)} | failed {failed} in {elapsed/60:.1f} min. Saved to {out_path}")


if __name__ == "__main__":
    main()


