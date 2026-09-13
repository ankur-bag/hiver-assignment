"""
Hiver Golden-Set Automated Evaluation Runner.
Executes end-to-end evaluation against /api/v1/chat/stream across 200 golden examples.
Parses SSE stream robustly, records metrics per example, and checkpoints immediately.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.compute_metrics import calculate_metrics_from_csv, print_console_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("golden_eval")

RESULT_COLUMNS = [
    "run_id",
    "example_id",
    "customer_text",
    "expected_intent",
    "predicted_intent",
    "intent_correct",
    "expected_escalate",
    "predicted_escalate",
    "escalation_correct",
    "expected_escalation_reason",
    "predicted_escalation_reason",
    "expected_reply_points",
    "generated_reply",
    "retrieved_context_available",
    "retrieved_context_count",
    "ttft_ms",
    "total_latency_ms",
    "difficulty",
    "language",
    "sampling_bucket",
    "request_status",
    "error_category",
    "error_type",
    "error_message",
    "timestamp",
    "golden_set_hash",
    "base_url",
]


def compute_file_hash(path: Path) -> str:
    """Computes a short SHA256 fingerprint of a file."""
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:12]


def parse_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes", "t")


async def parse_sse_stream(
    response: httpx.Response,
    request_start_time: float,
) -> Dict[str, Any]:
    """
    Robustly parses Server-Sent Events from an HTTP stream.
    Splits frames on double-newline boundaries, accumulates tokens, and extracts metadata.
    """
    buffer = ""
    tokens = []
    first_token_time: Optional[float] = None
    stream_complete_time: Optional[float] = None

    metadata_payload: Dict[str, Any] = {}
    grounding_payload: Dict[str, Any] = {}
    escalation_payload: Dict[str, Any] = {}
    complete_payload: Dict[str, Any] = {}
    error_payload: Optional[Dict[str, Any]] = None

    async for chunk in response.aiter_text():
        buffer += chunk
        parts = buffer.split("\n\n")
        buffer = parts.pop()  # Keep incomplete frame in buffer

        for part in parts:
            if not part.strip():
                continue

            event_type = "message"
            data_str = ""

            for line in part.split("\n"):
                line_trimmed = line.strip()
                if line_trimmed.startswith("event:"):
                    event_type = line_trimmed[6:].strip()
                elif line_trimmed.startswith("data:"):
                    data_str += line_trimmed[5:].strip()

            if not data_str:
                continue

            try:
                parsed_data = json.loads(data_str)
            except Exception:
                parsed_data = {"raw": data_str}

            if event_type == "token":
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                token_text = parsed_data.get("text", "")
                if token_text:
                    tokens.append(token_text)
            elif event_type == "metadata":
                metadata_payload = parsed_data
            elif event_type == "grounding":
                grounding_payload = parsed_data
            elif event_type == "escalation":
                escalation_payload = parsed_data
            elif event_type == "complete":
                complete_payload = parsed_data
                stream_complete_time = time.perf_counter()
            elif event_type == "error":
                error_payload = parsed_data

    # Calculate timings
    now = time.perf_counter()
    if stream_complete_time is None:
        stream_complete_time = now

    client_ttft_ms = (first_token_time - request_start_time) * 1000 if first_token_time else None
    client_total_ms = (stream_complete_time - request_start_time) * 1000

    # Prefer authoritative server telemetry if present
    telemetry = complete_payload.get("telemetry", {})
    ttft_ms = telemetry.get("ttft_ms", client_ttft_ms or client_total_ms)
    total_latency_ms = telemetry.get("total_time_ms", client_total_ms)

    generated_reply = "".join(tokens).strip()

    # Extract intent
    predicted_intent = (
        complete_payload.get("intent") or
        metadata_payload.get("intent") or
        ""
    )

    # Extract escalation
    predicted_escalate = (
        complete_payload.get("escalate")
        if "escalate" in complete_payload
        else escalation_payload.get("escalate", False)
    )
    predicted_escalation_reason = (
        complete_payload.get("escalation_reason") or
        escalation_payload.get("reason") or
        ""
    )

    # Extract grounding
    retrieved_count = (
        complete_payload.get("retrieved_cases") or
        grounding_payload.get("retrieved_context_count") or
        0
    )
    retrieved_avail = bool(
        complete_payload.get("retrieved_context") not in (None, "unavailable", 0, False) or
        grounding_payload.get("retrieved_context_available") or
        retrieved_count > 0
    )

    return {
        "generated_reply": generated_reply,
        "predicted_intent": predicted_intent,
        "predicted_escalate": bool(predicted_escalate),
        "predicted_escalation_reason": predicted_escalation_reason,
        "retrieved_context_available": retrieved_avail,
        "retrieved_context_count": retrieved_count,
        "ttft_ms": round(float(ttft_ms), 2) if ttft_ms is not None else None,
        "total_latency_ms": round(float(total_latency_ms), 2),
        "complete_received": bool(complete_payload),
        "error_payload": error_payload,
    }


async def evaluate_single_example(
    client: httpx.AsyncClient,
    base_url: str,
    row: Dict[str, Any],
    timeout_sec: float,
    run_id: str = "",
    golden_set_hash: str = "",
) -> Dict[str, Any]:
    """
    Sends one golden example to /api/v1/chat/stream and extracts evaluation results.
    Classifies errors into infrastructure, provider, or model failures.
    """
    example_id = row.get("example_id", "")
    customer_text = row.get("customer_text", "")
    expected_intent = row.get("expected_intent", "")
    expected_escalate = parse_bool(row.get("expected_escalate", False))
    expected_escalation_reason = row.get("expected_escalation_reason", "")
    expected_reply_points = row.get("expected_reply_points", "")
    language = row.get("language_final") or row.get("language_guess", "en")
    difficulty = row.get("difficulty_final") or row.get("difficulty_suggestion", "unknown")
    sampling_bucket = row.get("sampling_bucket", "")

    endpoint = f"{base_url.rstrip('/')}/api/v1/chat/stream"
    session_id = f"eval-{example_id}-{uuid.uuid4().hex[:6]}"

    payload = {
        "text": customer_text,
        "session_id": session_id,
        "language": language if language != "en" else None,
    }

    timestamp_str = datetime.now(timezone.utc).isoformat()

    result = {
        "run_id": run_id,
        "example_id": example_id,
        "customer_text": customer_text,
        "expected_intent": expected_intent,
        "predicted_intent": "",
        "intent_correct": False,
        "expected_escalate": expected_escalate,
        "predicted_escalate": False,
        "escalation_correct": False,
        "expected_escalation_reason": expected_escalation_reason,
        "predicted_escalation_reason": "",
        "expected_reply_points": expected_reply_points,
        "generated_reply": "",
        "retrieved_context_available": False,
        "retrieved_context_count": 0,
        "ttft_ms": 0.0,
        "total_latency_ms": 0.0,
        "difficulty": difficulty,
        "language": language,
        "sampling_bucket": sampling_bucket,
        "request_status": "FAILED",
        "error_category": "INFRASTRUCTURE",
        "error_type": "",
        "error_message": "",
        "timestamp": timestamp_str,
        "golden_set_hash": golden_set_hash,
        "base_url": base_url,
    }

    # Execute with at most 1 network retry for transient transport errors
    for attempt in range(2):
        t0 = time.perf_counter()
        try:
            async with client.stream(
                "POST",
                endpoint,
                json=payload,
                headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
                timeout=timeout_sec,
            ) as response:
                if response.status_code != 200:
                    err_body = await response.aread()
                    if response.status_code == 404:
                        result["error_category"] = "INFRASTRUCTURE"
                        result["error_type"] = "ENDPOINT_NOT_FOUND"
                        result["error_message"] = f"Endpoint 404 Not Found on {endpoint}"
                    elif response.status_code in (502, 503, 504):
                        result["error_category"] = "INFRASTRUCTURE"
                        result["error_type"] = f"HTTP_{response.status_code}"
                        result["error_message"] = err_body.decode("utf-8", errors="replace")[:300]
                    else:
                        result["error_category"] = "HTTP_ERROR"
                        result["error_type"] = f"HTTP_{response.status_code}"
                        result["error_message"] = err_body.decode("utf-8", errors="replace")[:300]
                    result["request_status"] = "FAILED"
                    return result

                parsed = await parse_sse_stream(response, request_start_time=t0)

                # Check if SSE stream emitted an error event
                if parsed.get("error_payload"):
                    err = parsed["error_payload"]
                    result["error_category"] = "PROVIDER"
                    result["error_type"] = err.get("code") or "PROVIDER_ERROR"
                    result["error_message"] = err.get("message") or err.get("error") or str(err)
                    result["request_status"] = "FAILED"
                    result["ttft_ms"] = parsed.get("ttft_ms") or 0.0
                    result["total_latency_ms"] = parsed.get("total_latency_ms") or 0.0
                    return result

                if not parsed.get("complete_received"):
                    result["error_category"] = "MODEL"
                    result["error_type"] = "INCOMPLETE_STREAM"
                    result["error_message"] = "Stream closed before complete event"
                    result["request_status"] = "FAILED"
                    result["generated_reply"] = parsed.get("generated_reply", "")
                    return result

                pred_intent = parsed.get("predicted_intent", "")
                pred_esc = parsed.get("predicted_escalate", False)

                result["predicted_intent"] = pred_intent
                result["intent_correct"] = bool(pred_intent and pred_intent == expected_intent)
                result["predicted_escalate"] = pred_esc
                result["escalation_correct"] = bool(pred_esc == expected_escalate)
                result["predicted_escalation_reason"] = parsed.get("predicted_escalation_reason", "")
                result["generated_reply"] = parsed.get("generated_reply", "")
                result["retrieved_context_available"] = parsed.get("retrieved_context_available", False)
                result["retrieved_context_count"] = parsed.get("retrieved_context_count", 0)
                result["ttft_ms"] = parsed.get("ttft_ms") or 0.0
                result["total_latency_ms"] = parsed.get("total_latency_ms") or 0.0
                result["request_status"] = "SUCCESS"
                result["error_category"] = "NONE"
                result["error_type"] = ""
                result["error_message"] = ""
                return result

        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.TimeoutException) as exc:
            if attempt == 1:
                result["error_category"] = "MODEL" if isinstance(exc, httpx.ReadTimeout) else "INFRASTRUCTURE"
                result["error_type"] = "TIMEOUT"
                result["error_message"] = str(exc)
                result["request_status"] = "FAILED"
                return result
            await asyncio.sleep(0.5)
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            if attempt == 1:
                result["error_category"] = "INFRASTRUCTURE"
                result["error_type"] = "CONNECT_ERROR"
                result["error_message"] = str(exc)
                result["request_status"] = "FAILED"
                return result
            await asyncio.sleep(0.5)
        except Exception as exc:
            result["error_category"] = "EVAL"
            result["error_type"] = "EVAL_EXCEPTION"
            result["error_message"] = str(exc)
            result["request_status"] = "FAILED"
            return result

    return result


def load_existing_results(output_file: Path) -> Dict[str, Dict[str, Any]]:
    """
    Loads existing evaluation results from CSV, keyed by example_id.
    Defensively deduplicates by keeping the latest record per example_id.
    """
    existing: Dict[str, Dict[str, Any]] = {}
    if not output_file.exists() or output_file.stat().st_size == 0:
        return existing

    with open(output_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            eid = r.get("example_id")
            if eid:
                existing[eid] = r
    return existing


def save_results_atomically(
    output_file: Path,
    results_by_id: Dict[str, Dict[str, Any]],
    ordered_example_ids: Optional[List[str]] = None,
):
    """
    Atomically writes results_by_id to output_file with deduplication and deterministic ordering.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = output_file.with_name(f"{output_file.name}.tmp.{uuid.uuid4().hex[:8]}")
    order = ordered_example_ids or list(results_by_id.keys())
    written_ids = set()

    try:
        with open(temp_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
            writer.writeheader()

            # Write in deterministic order
            for eid in order:
                if eid in results_by_id:
                    writer.writerow(results_by_id[eid])
                    written_ids.add(eid)

            # Write any IDs not in the provided order
            for eid, record in results_by_id.items():
                if eid not in written_ids:
                    writer.writerow(record)
                    written_ids.add(eid)

            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_file, output_file)
    finally:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass


async def run_evaluation(
    base_url: str,
    input_file: Path,
    output_file: Path,
    summary_file: Path,
    concurrency: int = 3,
    limit: Optional[int] = None,
    resume: bool = False,
    retry_errors: bool = False,
    timeout_sec: float = 40.0,
    consecutive_infra_threshold: int = 2,
):
    if not input_file.exists():
        raise FileNotFoundError(f"Input golden set CSV not found: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    # Compute golden set hash
    golden_hash = compute_file_hash(input_file)
    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    # Read input examples
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    if limit:
        all_rows = all_rows[:limit]

    logger.info("Loaded %d golden evaluation examples from %s (hash: %s)", len(all_rows), input_file, golden_hash)
    golden_map = {r["example_id"]: r for r in all_rows if r.get("example_id")}
    ordered_ids = [r["example_id"] for r in all_rows if r.get("example_id")]
    total_golden = len(ordered_ids)

    # Check existing results for resume / retry
    completed_ids = set()
    failed_ids = set()
    results_by_id: Dict[str, Dict[str, Any]] = {}

    if (resume or retry_errors) and output_file.exists():
        raw_existing = load_existing_results(output_file)

        # Drop old IDs absent from current golden file & normalize with current golden labels
        for eid, r in raw_existing.items():
            if eid not in golden_map:
                logger.info("Dropping stale example_id not in current golden set: %s", eid)
                continue

            golden_row = golden_map[eid]
            r["customer_text"] = golden_row.get("customer_text", r.get("customer_text", ""))
            r["expected_intent"] = golden_row.get("expected_intent", r.get("expected_intent", ""))
            r["expected_escalate"] = parse_bool(golden_row.get("expected_escalate", r.get("expected_escalate", False)))
            r["expected_escalation_reason"] = golden_row.get("expected_escalation_reason", r.get("expected_escalation_reason", ""))
            r["expected_reply_points"] = golden_row.get("expected_reply_points", r.get("expected_reply_points", ""))
            r["difficulty"] = golden_row.get("difficulty_final") or golden_row.get("difficulty_suggestion", r.get("difficulty", "unknown"))
            r["language"] = golden_row.get("language_final") or golden_row.get("language_guess", r.get("language", "en"))
            r["sampling_bucket"] = golden_row.get("sampling_bucket", r.get("sampling_bucket", ""))

            # Re-evaluate correctness flags against current golden labels
            pred_intent = r.get("predicted_intent", "")
            if pred_intent:
                r["intent_correct"] = str(bool(pred_intent == r["expected_intent"]))
            if "predicted_escalate" in r:
                r["escalation_correct"] = str(bool(parse_bool(r.get("predicted_escalate", False)) == parse_bool(r["expected_escalate"])))

            results_by_id[eid] = r
            if r.get("request_status") == "SUCCESS":
                completed_ids.add(eid)
            else:
                failed_ids.add(eid)

        logger.info(
            "Resume mode: Found %d unique valid records (%d success, %d failed)",
            len(results_by_id), len(completed_ids), len(failed_ids)
        )

        # Write normalized snapshot to disk before execution
        save_results_atomically(output_file, results_by_id, ordered_example_ids=ordered_ids)
    else:
        # If --resume is NOT supplied: start a genuinely fresh result snapshot
        results_by_id = {}
        save_results_atomically(output_file, results_by_id, ordered_example_ids=ordered_ids)

    # Determine rows to process
    rows_to_process = []
    for r in all_rows:
        eid = r.get("example_id")
        if retry_errors and eid in failed_ids:
            rows_to_process.append(r)
        elif resume and eid in completed_ids:
            continue
        elif resume and not retry_errors and eid in results_by_id:
            continue
        elif not retry_errors:
            rows_to_process.append(r)

    logger.info("Queued %d examples for execution (concurrency: %d)", len(rows_to_process), concurrency)

    if not rows_to_process and output_file.exists():
        logger.info("All examples already processed. Computing final metrics...")
        assert len(results_by_id) <= total_golden, f"Invariant violation: results ({len(results_by_id)}) > golden set ({total_golden})"
        assert len(results_by_id) <= 200, f"Invariant violation: results ({len(results_by_id)}) > 200"
        summary = calculate_metrics_from_csv(output_file, max_expected_examples=total_golden)
        summary["run_id"] = run_id
        summary["golden_set_hash"] = golden_hash
        summary["base_url"] = base_url
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print_console_report(summary, result_file=str(output_file))
        return

    file_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(concurrency)
    progress_count = 0
    total_to_run = len(rows_to_process)

    consecutive_infra_failures = 0
    circuit_breaker_triggered = False

    async def worker(row: Dict[str, Any], client: httpx.AsyncClient):
        nonlocal progress_count, consecutive_infra_failures, circuit_breaker_triggered

        if circuit_breaker_triggered:
            return

        async with semaphore:
            if circuit_breaker_triggered:
                return

            eval_result = await evaluate_single_example(
                client=client,
                base_url=base_url,
                row=row,
                timeout_sec=timeout_sec,
                run_id=run_id,
                golden_set_hash=golden_hash,
            )

            # Checkpoint row immediately and atomically
            async with file_lock:
                results_by_id[eval_result["example_id"]] = eval_result
                save_results_atomically(
                    output_file=output_file,
                    results_by_id=results_by_id,
                    ordered_example_ids=ordered_ids,
                )

                # Infrastructure failure circuit breaker check
                if eval_result.get("error_category") == "INFRASTRUCTURE":
                    consecutive_infra_failures += 1
                    if consecutive_infra_failures >= consecutive_infra_threshold:
                        circuit_breaker_triggered = True
                        logger.error(
                            "Evaluation stopped: /api/v1/chat/stream is no longer available "
                            "(%d consecutive infrastructure errors). Check the local backend process/router before resuming.",
                            consecutive_infra_failures
                        )
                        print(
                            f"\nEvaluation stopped: /api/v1/chat/stream is no longer available.\n"
                            f"Check the local backend process/router before resuming.\n"
                        )
                elif eval_result.get("request_status") == "SUCCESS":
                    consecutive_infra_failures = 0

            progress_count += 1
            status_symbol = "✓" if eval_result["request_status"] == "SUCCESS" else "✗"
            intent_symbol = "✓" if eval_result["intent_correct"] else "✗"
            logger.info(
                "[%d/%d] %s (ID: %s) -> Status: %s | Intent: %s (%s vs %s) | TTFT: %.0fms | Total: %.0fms",
                progress_count,
                total_to_run,
                status_symbol,
                eval_result["example_id"],
                eval_result["request_status"],
                intent_symbol,
                eval_result["predicted_intent"] or "NONE",
                eval_result["expected_intent"],
                eval_result["ttft_ms"] or 0,
                eval_result["total_latency_ms"] or 0,
            )

    # Run tasks with connection pooling
    limits = httpx.Limits(max_keepalive_connections=concurrency + 2, max_connections=concurrency + 5)
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = [worker(row, client) for row in rows_to_process]
        await asyncio.gather(*tasks)

    # Enforce invariants before publishing metrics
    assert len(results_by_id) <= total_golden, f"Invariant violation: unique result count ({len(results_by_id)}) > golden set ({total_golden})"
    assert len(results_by_id) <= 200, f"Invariant violation: unique result count ({len(results_by_id)}) > 200"

    logger.info("Evaluation run completed. Computing metrics across %s...", output_file)
    summary = calculate_metrics_from_csv(output_file, max_expected_examples=total_golden)
    summary["run_id"] = run_id
    summary["golden_set_hash"] = golden_hash
    summary["base_url"] = base_url
    if circuit_breaker_triggered:
        summary["circuit_breaker_triggered"] = True

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_console_report(summary, result_file=str(output_file))


def main():
    parser = argparse.ArgumentParser(description="Hiver Golden-Set Automated Evaluation Runner")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL for support API (e.g. http://127.0.0.1:8000 or Cloudflare Worker URL)",
    )
    parser.add_argument(
        "--input",
        default="evaluation/golden_set_200.csv",
        help="Path to input golden set CSV",
    )
    parser.add_argument(
        "--output",
        default="evaluation/results/golden_eval_results.csv",
        help="Path to save per-example result CSV",
    )
    parser.add_argument(
        "--summary-output",
        default="evaluation/results/golden_eval_summary.json",
        help="Path to save summary JSON metrics",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Number of concurrent requests (default: 3, allowed: 1-5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of examples to run",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing result CSV and skip completed examples",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry only failed rows from existing results CSV",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=40.0,
        help="Timeout in seconds for each SSE stream connection",
    )

    args = parser.parse_args()

    concurrency = max(1, min(5, args.concurrency))

    asyncio.run(
        run_evaluation(
            base_url=args.base_url,
            input_file=Path(args.input),
            output_file=Path(args.output),
            summary_file=Path(args.summary_output),
            concurrency=concurrency,
            limit=args.limit,
            resume=args.resume,
            retry_errors=args.retry_errors,
            timeout_sec=args.timeout,
        )
    )


if __name__ == "__main__":
    main()

