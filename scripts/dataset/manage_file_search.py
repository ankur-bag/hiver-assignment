"""
Manage Gemini File Search store creation, resumable shard ingestion, and verification.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv

# Resolve repository root
ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / "backend" / ".env"

# Load backend/.env safely
load_dotenv(ENV_PATH)

QUERIES = (
    "Where is my refund?",
    "My package has not arrived yet",
    "I cannot access my account",
    "My order is delayed",
    "Mera parcel abhi tak nahi aaya",
    "Mi pedido todavía no ha llegado",
)

STORE_DISPLAY_NAME = "hiver-amazon-support"


def get_api_key() -> str:
    """Validates and returns the GEMINI_API_KEY without exposing its contents."""
    key = os.getenv("GEMINI_API_KEY")
    if not key or not key.strip():
        print("ERROR: GEMINI_API_KEY is not configured in backend/.env or environment.")
        sys.exit(1)
    return key.strip()


def get_gemini_client(api_key: str):
    """Initializes Google GenAI client."""
    from google import genai
    return genai.Client(api_key=api_key)


def find_or_create_store(client, explicit_store_name: Optional[str] = None, force_create: bool = False) -> str:
    """
    Finds an existing File Search store matching STORE_DISPLAY_NAME, or creates one if needed.
    """
    from google.genai import types

    if explicit_store_name:
        try:
            store = client.file_search_stores.get(name=explicit_store_name)
            print(f"Found existing store by name: {store.name} (display_name: '{getattr(store, 'display_name', '')}')")
            return store.name
        except Exception as exc:
            print(f"Warning: Specified store '{explicit_store_name}' could not be fetched: {exc}")

    # Search existing stores by display name
    print(f"Searching for existing store with display name: '{STORE_DISPLAY_NAME}'...")
    try:
        stores_pager = client.file_search_stores.list()
        for store in stores_pager:
            dname = getattr(store, "display_name", None) or getattr(store, "displayName", None)
            if dname == STORE_DISPLAY_NAME:
                print(f"Reusing existing store: name={store.name}, display_name='{dname}'")
                return store.name
    except Exception as exc:
        print(f"Notice: Could not list existing stores ({exc}). Proceeding...")

    if not force_create:
        # Check if environment already had one
        env_store = os.getenv("GEMINI_FILE_SEARCH_STORE")
        if env_store:
            print(f"Reusing GEMINI_FILE_SEARCH_STORE from environment: {env_store}")
            return env_store

    print(f"Creating new File Search store with display name '{STORE_DISPLAY_NAME}'...")
    store = client.file_search_stores.create(
        config=types.CreateFileSearchStoreConfig(display_name=STORE_DISPLAY_NAME)
    )
    print(f"Created File Search store: name={store.name}, display_name='{getattr(store, 'display_name', STORE_DISPLAY_NAME)}'")
    return store.name


def persist_store_name_to_env(store_name: str):
    """Idempotently updates or appends GEMINI_FILE_SEARCH_STORE in backend/.env."""
    print(f"\nGEMINI_FILE_SEARCH_STORE={store_name}")
    if not ENV_PATH.exists():
        with open(ENV_PATH, "w", encoding="utf-8") as f:
            f.write(f"GEMINI_FILE_SEARCH_STORE={store_name}\n")
        print(f"Written GEMINI_FILE_SEARCH_STORE to {ENV_PATH}")
        return

    content = ENV_PATH.read_text(encoding="utf-8")
    lines = content.splitlines()
    new_lines = []
    found = False

    for line in lines:
        if line.strip().startswith("GEMINI_FILE_SEARCH_STORE="):
            new_lines.append(f"GEMINI_FILE_SEARCH_STORE={store_name}")
            found = True
        else:
            new_lines.append(line)

    if not found:
        new_lines.append(f"GEMINI_FILE_SEARCH_STORE={store_name}")

    ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"Persisted GEMINI_FILE_SEARCH_STORE to {ENV_PATH}")


def wait_for_operation(client, operation, timeout: int = 600):
    """Polls an async operation until completion with timeout."""
    deadline = time.monotonic() + timeout
    while not operation.done:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Operation {operation.name} timed out after {timeout}s")
        time.sleep(2)
        operation = client.operations.get(operation)
    if operation.error:
        raise RuntimeError(f"Operation failed with error: {operation.error}")
    return operation


def load_checkpoint(checkpoint_file: Path) -> set[str]:
    """Loads indexed shard filenames from JSON checkpoint."""
    if checkpoint_file.exists():
        try:
            data = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            return set(data.get("completed_shards", []))
        except Exception as exc:
            print(f"Warning: could not read checkpoint file: {exc}")
    return set()


def save_checkpoint(checkpoint_file: Path, completed: set[str]):
    """Saves completed shard set to JSON checkpoint."""
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed_shards": sorted(list(completed)),
        "count": len(completed),
    }
    checkpoint_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def upload_shards(client, store_name: str, dataset_dir: Path, checkpoint_file: Path):
    """Uploads and indexes text shards to File Search store with resumability."""
    from google.genai import types

    shards = sorted(dataset_dir.glob("*.txt"))
    if not shards:
        raise SystemExit(f"ERROR: No .txt shards found in {dataset_dir}")

    completed = load_checkpoint(checkpoint_file)
    total_shards = len(shards)
    print(f"\nFound {total_shards} shards in {dataset_dir}. Already completed: {len(completed)}.")

    successful = len(completed)
    failed = 0

    for idx, shard_path in enumerate(shards, 1):
        filename = shard_path.name
        if filename in completed:
            print(f"[{idx}/{total_shards}] {filename} -> already indexed (skipped).")
            continue

        print(f"[{idx}/{total_shards}] Uploading {filename} ({shard_path.stat().st_size:,} bytes)...")
        uploaded = False

        for attempt in range(1, 4):
            try:
                op = client.file_search_stores.upload_to_file_search_store(
                    file_search_store_name=store_name,
                    file=shard_path,
                    config=types.UploadToFileSearchStoreConfig(
                        mime_type="text/plain",
                        display_name=filename,
                    ),
                )
                print(f"  Indexing operation: {op.name} ... polling")
                finished_op = wait_for_operation(client, op)
                doc_name = getattr(getattr(finished_op, "response", None), "document_name", "ok")
                print(f"  Complete: document={doc_name}")
                completed.add(filename)
                save_checkpoint(checkpoint_file, completed)
                successful += 1
                uploaded = True
                break
            except Exception as exc:
                err_str = str(exc)
                print(f"  Attempt {attempt}/3 failed for {filename}: {exc}")
                if "429" in err_str or "quota" in err_str.lower() or "resource_exhausted" in err_str.lower():
                    time.sleep(5.0 * attempt)
                else:
                    time.sleep(2.0 * attempt)

        if not uploaded:
            print(f"ERROR: Failed to upload shard {filename} after 3 attempts.")
            failed += 1

    print("\n" + "=" * 50)
    print("INGESTION SUMMARY:")
    print(f"  Total shards:      {total_shards}")
    print(f"  Successful shards: {successful}")
    print(f"  Failed shards:     {failed}")
    print("=" * 50)

    try:
        store_status = client.file_search_stores.get(name=store_name)
        pending = getattr(store_status, "pending_documents_count", 0) or 0
        failed_docs = getattr(store_status, "failed_documents_count", 0) or 0
        print(f"Store status: active documents pending={pending}, failed={failed_docs}")
    except Exception as exc:
        print(f"Could not retrieve store status: {exc}")


def verify_file_search(client, store_name: str, model_name: str):
    """Runs test queries using File Search tool and verifies grounding."""
    from google.genai import types

    print("\n" + "=" * 50)
    print("VERIFYING FILE SEARCH RETRIEVAL & GROUNDING")
    print("=" * 50)

    tool = types.Tool(file_search=types.FileSearch(file_search_store_names=[store_name], top_k=5))
    config = types.GenerateContentConfig(tools=[tool], temperature=0.2)

    for query in QUERIES:
        print(f"\nQuery: {query}")
        try:
            res = client.models.generate_content(
                model=model_name,
                contents=query,
                config=config,
            )
            grounding_chunks = []
            candidates = getattr(res, "candidates", None) or []
            if candidates:
                gm = getattr(candidates[0], "grounding_metadata", None)
                for chunk in (getattr(gm, "grounding_chunks", None) or []):
                    rc = getattr(chunk, "retrieved_context", None)
                    if rc:
                        grounding_chunks.append({
                            "title": getattr(rc, "title", None),
                            "uri": getattr(rc, "uri", None),
                        })

            safe_text = res.text.encode("utf-8", errors="replace").decode("utf-8")
            print(f"Answer: {safe_text[:200]}...")
            print(f"Grounding Chunks Retrieved: {len(grounding_chunks)}")
            for idx, gc in enumerate(grounding_chunks[:3], 1):
                print(f"  [{idx}] {gc}")
        except Exception as exc:
            print(f"Verification error for query: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Manage Gemini File Search store.")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "file_search_dataset")
    parser.add_argument("--checkpoint-file", type=Path, default=ROOT / "file_search_dataset" / "upload_checkpoint.json")
    parser.add_argument("--store", default=os.getenv("GEMINI_FILE_SEARCH_STORE"))
    parser.add_argument("--create", action="store_true", help="Create or reuse File Search store")
    parser.add_argument("--upload", action="store_true", help="Upload dataset shards")
    parser.add_argument("--verify", action="store_true", help="Verify search grounding")
    parser.add_argument("--preflight", action="store_true", help="Verify environment and key only")
    parser.add_argument("--model", default=os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash"))

    args = parser.parse_args()

    api_key = get_api_key()
    print("Preflight: GEMINI_API_KEY loaded = True")

    if args.preflight and not (args.create or args.upload or args.verify):
        return

    client = get_gemini_client(api_key)

    store_name = args.store
    if args.create or not store_name:
        store_name = find_or_create_store(client, explicit_store_name=args.store, force_create=args.create)
        persist_store_name_to_env(store_name)

    if args.upload:
        if not store_name:
            print("ERROR: store_name is required for upload.")
            sys.exit(1)
        upload_shards(client, store_name, args.dataset_dir, args.checkpoint_file)

    if args.verify:
        if not store_name:
            print("ERROR: store_name is required for verify.")
            sys.exit(1)
        verify_file_search(client, store_name, args.model)


if __name__ == "__main__":
    main()
