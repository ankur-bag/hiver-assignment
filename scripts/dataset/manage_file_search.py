"""Create, populate, poll, and smoke-test a Gemini File Search store."""

import argparse
import json
import os
import time
from pathlib import Path

from google import genai
from google.genai import types

QUERIES = (
    "My package has not arrived yet", "Where is my refund?", "I cannot access my account",
    "My order is delayed", "Mera parcel abhi tak nahi aaya", "Mi pedido todavía no ha llegado",
)


def wait(client, operation, timeout=900):
    deadline = time.monotonic() + timeout
    while not operation.done:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Operation {operation.name} did not finish within {timeout}s")
        time.sleep(2)
        operation = client.operations.get(operation)
    if operation.error:
        raise RuntimeError(f"Operation {operation.name} failed: {operation.error}")
    return operation


def upload_with_retry(client, store_name, path):
    for attempt in range(3):
        try:
            operation = client.file_search_stores.upload_to_file_search_store(
                file_search_store_name=store_name, file=path,
                config=types.UploadToFileSearchStoreConfig(mime_type="text/plain", display_name=path.name),
            )
            return wait(client, operation)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def grounding(response):
    result = []
    for candidate in response.candidates or []:
        for chunk in (getattr(candidate.grounding_metadata, "grounding_chunks", None) or []):
            item = getattr(chunk, "retrieved_context", None)
            if item:
                result.append(item.model_dump(mode="json", exclude_none=True))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("file_search_dataset"))
    parser.add_argument("--store", default=os.getenv("GEMINI_FILE_SEARCH_STORE"))
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--model", default=os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash"))
    args = parser.parse_args()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is required")
    client = genai.Client(api_key=api_key)
    store_name = args.store
    if args.create:
        store = client.file_search_stores.create(config=types.CreateFileSearchStoreConfig(display_name="hiver-amazon-support"))
        store_name = store.name
        print(f"GEMINI_FILE_SEARCH_STORE={store_name}")
    if not store_name:
        raise SystemExit("Pass --store or set GEMINI_FILE_SEARCH_STORE")
    if args.upload:
        paths = sorted(args.dataset_dir.glob("*.txt"))
        if not paths:
            raise SystemExit(f"No .txt shards found in {args.dataset_dir}")
        for number, path in enumerate(paths, 1):
            operation = upload_with_retry(client, store_name, path)
            print(f"[{number}/{len(paths)}] indexed {path.name}: {operation.response.document_name}")
        store = client.file_search_stores.get(name=store_name)
        if store.pending_documents_count or store.failed_documents_count:
            raise RuntimeError(f"Store not ready: pending={store.pending_documents_count}, failed={store.failed_documents_count}")
        print(json.dumps(store.model_dump(mode="json"), indent=2))
    if args.verify:
        tool = types.Tool(file_search=types.FileSearch(file_search_store_names=[store_name], top_k=5))
        for query in QUERIES:
            response = client.models.generate_content(model=args.model, contents=query, config=types.GenerateContentConfig(tools=[tool]))
            print(json.dumps({"query": query, "answer": response.text, "grounding": grounding(response)}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("file_search_dataset"))
    parser.add_argument("--store", default=os.getenv("GEMINI_FILE_SEARCH_STORE"))
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    store_name = args.store
    if args.create:
        store_name = client.file_search_stores.create(config={"display_name": "hiver-amazon-support"}).name
        print(f"GEMINI_FILE_SEARCH_STORE={store_name}")
    if not store_name:
        raise SystemExit("Set GEMINI_FILE_SEARCH_STORE or pass --create")
    if args.upload:
        paths = sorted(args.dataset_dir.glob("*.txt"))
        if not paths:
            raise SystemExit(f"No .txt shards found in {args.dataset_dir}")
        for index, path in enumerate(paths, 1):
            upload_with_retry(client, store_name, path)
            print(f"Indexed {index}/{len(paths)}: {path.name}")
        store = client.file_search_stores.get(name=store_name)
        print(store.model_dump_json(indent=2, exclude_none=True))
        if store.pending_documents_count or store.failed_documents_count:
            raise SystemExit("Store is not ready: pending or failed documents remain")
    if args.verify:
        tool = types.Tool(file_search=types.FileSearch(file_search_store_names=[store_name], top_k=5))
        for query in QUERIES:
            response = client.models.generate_content(model=os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash"), contents=query, config={"tools": [tool]})
            print(json.dumps({"query": query, "answer": response.text, "grounding": grounding(response)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
