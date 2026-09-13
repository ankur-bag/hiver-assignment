import asyncio
import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from scripts.evaluation.compute_metrics import calculate_metrics_from_csv
from scripts.evaluation.run_golden_eval import (
    RESULT_COLUMNS,
    evaluate_single_example,
    load_existing_results,
    parse_sse_stream,
    run_evaluation,
    save_results_atomically,
)


class MockChunkResponse:
    def __init__(self, chunks: list[str], status_code: int = 200):
        self.chunks = chunks
        self.status_code = status_code

    async def aiter_text(self):
        for c in self.chunks:
            yield c

    async def aread(self):
        return "".join(self.chunks).encode("utf-8")


@pytest.mark.asyncio
async def test_sse_split_across_network_chunks():
    # Simulate a single SSE event split in half across 2 chunks
    chunks = [
        'event: status\ndata: {"stage": "analy',
        'zing"}\n\nevent: token\ndata: {"text": "Hello',
        ' world!"}\n\nevent: complete\ndata: {"intent": "ACCOUNT_ACCESS", "escalate": false, "retrieved_cases": 2}\n\n'
    ]
    resp = MockChunkResponse(chunks)
    parsed = await parse_sse_stream(resp, request_start_time=0.0)

    assert parsed["generated_reply"] == "Hello world!"
    assert parsed["predicted_intent"] == "ACCOUNT_ACCESS"
    assert parsed["predicted_escalate"] is False
    assert parsed["retrieved_context_count"] == 2
    assert parsed["complete_received"] is True


@pytest.mark.asyncio
async def test_multiple_events_in_one_network_chunk():
    chunk = (
        'event: status\ndata: {"stage": "retrieving"}\n\n'
        'event: token\ndata: {"text": "Part 1 "}\n\n'
        'event: token\ndata: {"text": "Part 2"}\n\n'
        'event: complete\ndata: {"intent": "PACKAGE_NOT_RECEIVED", "escalate": true, "escalation_reason": "Missing item"}\n\n'
    )
    resp = MockChunkResponse([chunk])
    parsed = await parse_sse_stream(resp, request_start_time=0.0)

    assert parsed["generated_reply"] == "Part 1 Part 2"
    assert parsed["predicted_intent"] == "PACKAGE_NOT_RECEIVED"
    assert parsed["predicted_escalate"] is True
    assert parsed["predicted_escalation_reason"] == "Missing item"
    assert parsed["complete_received"] is True


@pytest.mark.asyncio
async def test_missing_complete_event_handled():
    chunks = [
        'event: status\ndata: {"stage": "analyzing"}\n\n'
        'event: token\ndata: {"text": "Unfinished reply"}\n\n'
    ]
    resp = MockChunkResponse(chunks)
    parsed = await parse_sse_stream(resp, request_start_time=0.0)

    assert parsed["generated_reply"] == "Unfinished reply"
    assert parsed["complete_received"] is False


@pytest.mark.asyncio
async def test_provider_error_in_sse():
    chunks = [
        'event: error\ndata: {"code": "PROVIDER_QUOTA_EXHAUSTED", "message": "Daily quota exceeded"}\n\n'
    ]
    resp = MockChunkResponse(chunks)
    parsed = await parse_sse_stream(resp, request_start_time=0.0)

    assert parsed["error_payload"] is not None
    assert parsed["error_payload"]["code"] == "PROVIDER_QUOTA_EXHAUSTED"


def test_metric_computation_and_confusion_matrix():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "example_id", "expected_intent", "predicted_intent", "intent_correct",
            "expected_escalate", "predicted_escalate", "escalation_correct",
            "retrieved_context_available", "retrieved_context_count",
            "ttft_ms", "total_latency_ms", "difficulty", "language",
            "request_status", "error_type"
        ])
        writer.writeheader()
        # Row 1: correct intent, correct no-escalate, grounded
        writer.writerow({
            "example_id": "G001", "expected_intent": "ACCOUNT_ACCESS", "predicted_intent": "ACCOUNT_ACCESS",
            "intent_correct": "True", "expected_escalate": "FALSE", "predicted_escalate": "False",
            "escalation_correct": "True", "retrieved_context_available": "True", "retrieved_context_count": "2",
            "ttft_ms": "5000", "total_latency_ms": "6000", "difficulty": "easy", "language": "en",
            "request_status": "SUCCESS", "error_type": ""
        })
        # Row 2: incorrect intent, correct escalate, grounded
        writer.writerow({
            "example_id": "G002", "expected_intent": "REFUND_PENDING", "predicted_intent": "ORDER_STATUS",
            "intent_correct": "False", "expected_escalate": "TRUE", "predicted_escalate": "True",
            "escalation_correct": "True", "retrieved_context_available": "True", "retrieved_context_count": "3",
            "ttft_ms": "6000", "total_latency_ms": "7000", "difficulty": "medium", "language": "es",
            "request_status": "SUCCESS", "error_type": ""
        })
        # Row 3: failed request (should not distort latency)
        writer.writerow({
            "example_id": "G003", "expected_intent": "DELIVERY_DELAY", "predicted_intent": "",
            "intent_correct": "False", "expected_escalate": "FALSE", "predicted_escalate": "False",
            "escalation_correct": "False", "retrieved_context_available": "False", "retrieved_context_count": "0",
            "ttft_ms": "0", "total_latency_ms": "0", "difficulty": "hard", "language": "de",
            "request_status": "FAILED", "error_type": "TIMEOUT"
        })
        tmp_csv = f.name

    try:
        metrics = calculate_metrics_from_csv(tmp_csv)

        assert metrics["n_total"] == 3
        assert metrics["n_success"] == 2
        assert metrics["n_failed"] == 1
        assert metrics["intent"]["accuracy"] == 0.5  # 1 out of 2 successful
        assert metrics["escalation"]["accuracy"] == 1.0  # 2 out of 2 successful
        assert metrics["escalation"]["precision"] == 1.0
        assert metrics["escalation"]["recall"] == 1.0
        assert metrics["grounding"]["success_rate"] == 1.0  # 2 out of 2 grounded
        assert metrics["latency"]["median_ttft_ms"] == 5500.0
        assert metrics["latency"]["median_total_ms"] == 6500.0
        assert metrics["error_counts"]["TIMEOUT"] == 1

        # Check confusion matrix
        cm = metrics["intent"]["confusion_matrix"]
        assert cm["ACCOUNT_ACCESS"]["ACCOUNT_ACCESS"] == 1
        assert cm["REFUND_PENDING"]["ORDER_STATUS"] == 1
    finally:
        Path(tmp_csv).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_resume_and_no_duplicate_rows():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = Path(tmpdir) / "golden.csv"
        output_csv = Path(tmpdir) / "results.csv"
        summary_json = Path(tmpdir) / "summary.json"

        # Create 2 input rows
        with open(input_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "example_id", "customer_text", "expected_intent", "expected_escalate",
                "expected_escalation_reason", "expected_reply_points", "language_final", "difficulty_final", "sampling_bucket"
            ])
            writer.writeheader()
            writer.writerow({
                "example_id": "G001", "customer_text": "q1", "expected_intent": "ACCOUNT_ACCESS",
                "expected_escalate": "FALSE", "expected_escalation_reason": "none",
                "expected_reply_points": "reply", "language_final": "en", "difficulty_final": "easy", "sampling_bucket": "b1"
            })
            writer.writerow({
                "example_id": "G002", "customer_text": "q2", "expected_intent": "REFUND_PENDING",
                "expected_escalate": "FALSE", "expected_escalation_reason": "none",
                "expected_reply_points": "reply", "language_final": "en", "difficulty_final": "easy", "sampling_bucket": "b2"
            })

        # Pre-populate output CSV with G001 already successful
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            writer.writerow({
                "example_id": "G001", "customer_text": "q1", "expected_intent": "ACCOUNT_ACCESS",
                "predicted_intent": "ACCOUNT_ACCESS", "intent_correct": "True",
                "expected_escalate": "False", "predicted_escalate": "False", "escalation_correct": "True",
                "expected_escalation_reason": "none", "predicted_escalation_reason": "",
                "expected_reply_points": "reply", "generated_reply": "rep",
                "retrieved_context_available": "True", "retrieved_context_count": "1",
                "ttft_ms": "5000", "total_latency_ms": "6000", "difficulty": "easy",
                "language": "en", "sampling_bucket": "b1", "request_status": "SUCCESS",
                "error_type": "", "error_message": ""
            })

        # Mock single example evaluation for G002
        async def mock_evaluate_single(client, base_url, row, timeout_sec, **kwargs):
            return {
                "example_id": row["example_id"], "customer_text": row["customer_text"],
                "expected_intent": row["expected_intent"], "predicted_intent": "REFUND_PENDING",
                "intent_correct": True, "expected_escalate": False, "predicted_escalate": False,
                "escalation_correct": True, "expected_escalation_reason": "", "predicted_escalation_reason": "",
                "expected_reply_points": "", "generated_reply": "reply", "retrieved_context_available": True,
                "retrieved_context_count": 1, "ttft_ms": 5200.0, "total_latency_ms": 6100.0,
                "difficulty": "easy", "language": "en", "sampling_bucket": "b2",
                "request_status": "SUCCESS", "error_type": "", "error_message": ""
            }

        import scripts.evaluation.run_golden_eval as rge
        orig_evaluate = rge.evaluate_single_example
        rge.evaluate_single_example = mock_evaluate_single

        try:
            await run_evaluation(
                base_url="http://mock",
                input_file=input_csv,
                output_file=output_csv,
                summary_file=summary_json,
                concurrency=1,
                resume=True,
            )

            # Read result rows: G001 was skipped, G002 was added in order
            with open(output_csv, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 2
            assert rows[0]["example_id"] == "G001"
            assert rows[1]["example_id"] == "G002"
        finally:
            rge.evaluate_single_example = orig_evaluate


@pytest.mark.asyncio
async def test_failed_result_replaced_by_successful_retry():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = Path(tmpdir) / "golden.csv"
        output_csv = Path(tmpdir) / "results.csv"
        summary_json = Path(tmpdir) / "summary.json"

        # 3 golden examples
        with open(input_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "example_id", "customer_text", "expected_intent", "expected_escalate",
                "expected_escalation_reason", "expected_reply_points", "language_final", "difficulty_final", "sampling_bucket"
            ])
            writer.writeheader()
            for eid in ["G001", "G002", "G003"]:
                writer.writerow({
                    "example_id": eid, "customer_text": f"query {eid}", "expected_intent": "ACCOUNT_ACCESS",
                    "expected_escalate": "FALSE", "expected_escalation_reason": "none",
                    "expected_reply_points": "reply", "language_final": "en", "difficulty_final": "easy", "sampling_bucket": "b1"
                })

        # Pre-populate output CSV: G001 (SUCCESS), G002 (FAILED), G003 (SUCCESS)
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            writer.writerow({
                "example_id": "G001", "customer_text": "query G001", "expected_intent": "ACCOUNT_ACCESS",
                "predicted_intent": "ACCOUNT_ACCESS", "intent_correct": "True", "expected_escalate": "False",
                "predicted_escalate": "False", "escalation_correct": "True", "expected_escalation_reason": "",
                "predicted_escalation_reason": "", "expected_reply_points": "", "generated_reply": "ok",
                "retrieved_context_available": "True", "retrieved_context_count": "1", "ttft_ms": "5000",
                "total_latency_ms": "6000", "difficulty": "easy", "language": "en", "sampling_bucket": "b1",
                "request_status": "SUCCESS", "error_type": "", "error_message": ""
            })
            writer.writerow({
                "example_id": "G002", "customer_text": "query G002", "expected_intent": "ACCOUNT_ACCESS",
                "predicted_intent": "", "intent_correct": "False", "expected_escalate": "False",
                "predicted_escalate": "False", "escalation_correct": "False", "expected_escalation_reason": "",
                "predicted_escalation_reason": "", "expected_reply_points": "", "generated_reply": "",
                "retrieved_context_available": "False", "retrieved_context_count": "0", "ttft_ms": "0",
                "total_latency_ms": "0", "difficulty": "easy", "language": "en", "sampling_bucket": "b1",
                "request_status": "FAILED", "error_type": "HTTP_404", "error_message": "Not Found"
            })
            writer.writerow({
                "example_id": "G003", "customer_text": "query G003", "expected_intent": "ACCOUNT_ACCESS",
                "predicted_intent": "ACCOUNT_ACCESS", "intent_correct": "True", "expected_escalate": "False",
                "predicted_escalate": "False", "escalation_correct": "True", "expected_escalation_reason": "",
                "predicted_escalation_reason": "", "expected_reply_points": "", "generated_reply": "ok",
                "retrieved_context_available": "True", "retrieved_context_count": "1", "ttft_ms": "5100",
                "total_latency_ms": "6100", "difficulty": "easy", "language": "en", "sampling_bucket": "b1",
                "request_status": "SUCCESS", "error_type": "", "error_message": ""
            })

        evaluated_ids = []

        async def mock_evaluate_single(client, base_url, row, timeout_sec, **kwargs):
            evaluated_ids.append(row["example_id"])
            return {
                "example_id": row["example_id"], "customer_text": row["customer_text"],
                "expected_intent": row["expected_intent"], "predicted_intent": "ACCOUNT_ACCESS",
                "intent_correct": True, "expected_escalate": False, "predicted_escalate": False,
                "escalation_correct": True, "expected_escalation_reason": "", "predicted_escalation_reason": "",
                "expected_reply_points": "", "generated_reply": "fixed reply", "retrieved_context_available": True,
                "retrieved_context_count": 1, "ttft_ms": 4800.0, "total_latency_ms": 5500.0,
                "difficulty": "easy", "language": "en", "sampling_bucket": "b1",
                "request_status": "SUCCESS", "error_type": "", "error_message": ""
            }

        import scripts.evaluation.run_golden_eval as rge
        orig_evaluate = rge.evaluate_single_example
        rge.evaluate_single_example = mock_evaluate_single

        try:
            await run_evaluation(
                base_url="http://mock",
                input_file=input_csv,
                output_file=output_csv,
                summary_file=summary_json,
                concurrency=1,
                resume=True,
                retry_errors=True,
            )

            assert evaluated_ids == ["G002"]

            # Verify output file
            with open(output_csv, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            assert len(rows) == 3
            assert [r["example_id"] for r in rows] == ["G001", "G002", "G003"]
            assert rows[1]["example_id"] == "G002"
            assert rows[1]["request_status"] == "SUCCESS"
            assert rows[1]["generated_reply"] == "fixed reply"
        finally:
            rge.evaluate_single_example = orig_evaluate


@pytest.mark.asyncio
async def test_row_count_remains_constant_and_no_duplicates():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = Path(tmpdir) / "golden.csv"
        output_csv = Path(tmpdir) / "results.csv"
        summary_json = Path(tmpdir) / "summary.json"

        # 5 golden examples
        with open(input_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "example_id", "customer_text", "expected_intent", "expected_escalate",
                "expected_escalation_reason", "expected_reply_points", "language_final", "difficulty_final", "sampling_bucket"
            ])
            writer.writeheader()
            for i in range(1, 6):
                eid = f"G00{i}"
                writer.writerow({
                    "example_id": eid, "customer_text": f"text {eid}", "expected_intent": "ACCOUNT_ACCESS",
                    "expected_escalate": "FALSE", "expected_escalation_reason": "none",
                    "expected_reply_points": "reply", "language_final": "en", "difficulty_final": "easy", "sampling_bucket": "b1"
                })

        # Pre-populate 5 rows: G001, G004, G005 SUCCESS; G002, G003 FAILED
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            for i in range(1, 6):
                eid = f"G00{i}"
                status = "FAILED" if eid in ("G002", "G003") else "SUCCESS"
                writer.writerow({
                    "example_id": eid, "customer_text": f"text {eid}", "expected_intent": "ACCOUNT_ACCESS",
                    "predicted_intent": "ACCOUNT_ACCESS" if status == "SUCCESS" else "",
                    "intent_correct": "True" if status == "SUCCESS" else "False", "expected_escalate": "False",
                    "predicted_escalate": "False", "escalation_correct": "True", "expected_escalation_reason": "",
                    "predicted_escalation_reason": "", "expected_reply_points": "",
                    "generated_reply": "ok" if status == "SUCCESS" else "",
                    "retrieved_context_available": "True" if status == "SUCCESS" else "False",
                    "retrieved_context_count": "1" if status == "SUCCESS" else "0",
                    "ttft_ms": "5000" if status == "SUCCESS" else "0",
                    "total_latency_ms": "6000" if status == "SUCCESS" else "0",
                    "difficulty": "easy", "language": "en", "sampling_bucket": "b1",
                    "request_status": status, "error_type": "" if status == "SUCCESS" else "TIMEOUT",
                    "error_message": ""
                })

        # Run retry-errors
        async def mock_evaluate_single(client, base_url, row, timeout_sec, **kwargs):
            return {
                "example_id": row["example_id"], "customer_text": row["customer_text"],
                "expected_intent": row["expected_intent"], "predicted_intent": "ACCOUNT_ACCESS",
                "intent_correct": True, "expected_escalate": False, "predicted_escalate": False,
                "escalation_correct": True, "expected_escalation_reason": "", "predicted_escalation_reason": "",
                "expected_reply_points": "", "generated_reply": "retried", "retrieved_context_available": True,
                "retrieved_context_count": 1, "ttft_ms": 4500.0, "total_latency_ms": 5200.0,
                "difficulty": "easy", "language": "en", "sampling_bucket": "b1",
                "request_status": "SUCCESS", "error_type": "", "error_message": ""
            }

        import scripts.evaluation.run_golden_eval as rge
        orig_evaluate = rge.evaluate_single_example
        rge.evaluate_single_example = mock_evaluate_single

        try:
            await run_evaluation(
                base_url="http://mock",
                input_file=input_csv,
                output_file=output_csv,
                summary_file=summary_json,
                concurrency=2,
                resume=True,
                retry_errors=True,
            )

            with open(output_csv, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            assert len(rows) == 5
            example_ids = [r["example_id"] for r in rows]
            assert len(set(example_ids)) == 5
            assert all(r["request_status"] == "SUCCESS" for r in rows)
        finally:
            rge.evaluate_single_example = orig_evaluate


def test_metrics_operate_on_unique_example_ids():
    # Simulate a CSV with 207 rows (200 unique IDs, 7 failed records followed by 7 retried successful records)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()

        # Write 200 initial records (193 SUCCESS, 7 FAILED)
        for i in range(1, 201):
            eid = f"G{i:03d}"
            status = "FAILED" if i > 193 else "SUCCESS"
            writer.writerow({
                "example_id": eid, "customer_text": f"text {eid}", "expected_intent": "ACCOUNT_ACCESS",
                "predicted_intent": "ACCOUNT_ACCESS" if status == "SUCCESS" else "",
                "intent_correct": "True" if status == "SUCCESS" else "False", "expected_escalate": "False",
                "predicted_escalate": "False", "escalation_correct": "True", "expected_escalation_reason": "",
                "predicted_escalation_reason": "", "expected_reply_points": "",
                "generated_reply": "ok" if status == "SUCCESS" else "",
                "retrieved_context_available": "True" if status == "SUCCESS" else "False",
                "retrieved_context_count": "1" if status == "SUCCESS" else "0",
                "ttft_ms": "5000" if status == "SUCCESS" else "0",
                "total_latency_ms": "6000" if status == "SUCCESS" else "0",
                "difficulty": "easy", "language": "en", "sampling_bucket": "b1",
                "request_status": status, "error_type": "" if status == "SUCCESS" else "HTTP_404",
                "error_message": ""
            })

        # Append 7 retried SUCCESS records for G194..G200 (simulating the old buggy append behavior)
        for i in range(194, 201):
            eid = f"G{i:03d}"
            writer.writerow({
                "example_id": eid, "customer_text": f"text {eid}", "expected_intent": "ACCOUNT_ACCESS",
                "predicted_intent": "ACCOUNT_ACCESS", "intent_correct": "True", "expected_escalate": "False",
                "predicted_escalate": "False", "escalation_correct": "True", "expected_escalation_reason": "",
                "predicted_escalation_reason": "", "expected_reply_points": "",
                "generated_reply": "ok retry", "retrieved_context_available": "True",
                "retrieved_context_count": "1", "ttft_ms": "5100", "total_latency_ms": "6100",
                "difficulty": "easy", "language": "en", "sampling_bucket": "b1",
                "request_status": "SUCCESS", "error_type": "", "error_message": ""
            })

        tmp_csv = f.name

    try:
        metrics = calculate_metrics_from_csv(tmp_csv, max_expected_examples=None, strict_no_duplicates=False)
        assert metrics["n_total"] == 200
        assert metrics["n_success"] == 200
        assert metrics["n_failed"] == 0
        assert len(metrics["error_counts"]) == 0
    finally:
        Path(tmp_csv).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_resume_sees_updated_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = Path(tmpdir) / "golden.csv"
        output_csv = Path(tmpdir) / "results.csv"
        summary_json = Path(tmpdir) / "summary.json"

        with open(input_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "example_id", "customer_text", "expected_intent", "expected_escalate",
                "expected_escalation_reason", "expected_reply_points", "language_final", "difficulty_final", "sampling_bucket"
            ])
            writer.writeheader()
            writer.writerow({
                "example_id": "G001", "customer_text": "q1", "expected_intent": "ACCOUNT_ACCESS",
                "expected_escalate": "FALSE", "expected_escalation_reason": "none",
                "expected_reply_points": "reply", "language_final": "en", "difficulty_final": "easy", "sampling_bucket": "b1"
            })

        # Initially failed
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            writer.writerow({
                "example_id": "G001", "customer_text": "q1", "expected_intent": "ACCOUNT_ACCESS",
                "predicted_intent": "", "intent_correct": "False", "expected_escalate": "False",
                "predicted_escalate": "False", "escalation_correct": "False", "expected_escalation_reason": "",
                "predicted_escalation_reason": "", "expected_reply_points": "", "generated_reply": "",
                "retrieved_context_available": "False", "retrieved_context_count": "0", "ttft_ms": "0",
                "total_latency_ms": "0", "difficulty": "easy", "language": "en", "sampling_bucket": "b1",
                "request_status": "FAILED", "error_type": "HTTP_500", "error_message": ""
            })

        eval_calls = 0

        async def mock_evaluate_single(client, base_url, row, timeout_sec, **kwargs):
            nonlocal eval_calls
            eval_calls += 1
            return {
                "example_id": row["example_id"], "customer_text": row["customer_text"],
                "expected_intent": row["expected_intent"], "predicted_intent": "ACCOUNT_ACCESS",
                "intent_correct": True, "expected_escalate": False, "predicted_escalate": False,
                "escalation_correct": True, "expected_escalation_reason": "", "predicted_escalation_reason": "",
                "expected_reply_points": "", "generated_reply": "fixed", "retrieved_context_available": True,
                "retrieved_context_count": 1, "ttft_ms": 5000.0, "total_latency_ms": 6000.0,
                "difficulty": "easy", "language": "en", "sampling_bucket": "b1",
                "request_status": "SUCCESS", "error_type": "", "error_message": ""
            }

        import scripts.evaluation.run_golden_eval as rge
        orig_evaluate = rge.evaluate_single_example
        rge.evaluate_single_example = mock_evaluate_single

        try:
            # First retry
            await run_evaluation(
                base_url="http://mock",
                input_file=input_csv,
                output_file=output_csv,
                summary_file=summary_json,
                resume=True,
                retry_errors=True,
            )
            assert eval_calls == 1

            # Second run with --resume (should skip G001 now that it's SUCCESS)
            await run_evaluation(
                base_url="http://mock",
                input_file=input_csv,
                output_file=output_csv,
                summary_file=summary_json,
                resume=True,
            )
            assert eval_calls == 1  # No additional calls!
        finally:
            rge.evaluate_single_example = orig_evaluate


@pytest.mark.asyncio
async def test_retry_errors_no_longer_retains_stale_failure():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = Path(tmpdir) / "golden.csv"
        output_csv = Path(tmpdir) / "results.csv"
        summary_json = Path(tmpdir) / "summary.json"

        with open(input_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "example_id", "customer_text", "expected_intent", "expected_escalate",
                "expected_escalation_reason", "expected_reply_points", "language_final", "difficulty_final", "sampling_bucket"
            ])
            writer.writeheader()
            writer.writerow({
                "example_id": "G001", "customer_text": "q1", "expected_intent": "ACCOUNT_ACCESS",
                "expected_escalate": "FALSE", "expected_escalation_reason": "none",
                "expected_reply_points": "reply", "language_final": "en", "difficulty_final": "easy", "sampling_bucket": "b1"
            })

        # Initially failed
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            writer.writerow({
                "example_id": "G001", "customer_text": "q1", "expected_intent": "ACCOUNT_ACCESS",
                "predicted_intent": "", "intent_correct": "False", "expected_escalate": "False",
                "predicted_escalate": "False", "escalation_correct": "False", "expected_escalation_reason": "",
                "predicted_escalation_reason": "", "expected_reply_points": "", "generated_reply": "",
                "retrieved_context_available": "False", "retrieved_context_count": "0", "ttft_ms": "0",
                "total_latency_ms": "0", "difficulty": "easy", "language": "en", "sampling_bucket": "b1",
                "request_status": "FAILED", "error_type": "HTTP_404", "error_message": ""
            })

        eval_calls = 0

        async def mock_evaluate_single(client, base_url, row, timeout_sec, **kwargs):
            nonlocal eval_calls
            eval_calls += 1
            return {
                "example_id": row["example_id"], "customer_text": row["customer_text"],
                "expected_intent": row["expected_intent"], "predicted_intent": "ACCOUNT_ACCESS",
                "intent_correct": True, "expected_escalate": False, "predicted_escalate": False,
                "escalation_correct": True, "expected_escalation_reason": "", "predicted_escalation_reason": "",
                "expected_reply_points": "", "generated_reply": "fixed", "retrieved_context_available": True,
                "retrieved_context_count": 1, "ttft_ms": 5000.0, "total_latency_ms": 6000.0,
                "difficulty": "easy", "language": "en", "sampling_bucket": "b1",
                "request_status": "SUCCESS", "error_type": "", "error_message": ""
            }

        import scripts.evaluation.run_golden_eval as rge
        orig_evaluate = rge.evaluate_single_example
        rge.evaluate_single_example = mock_evaluate_single

        try:
            # Retry errors
            await run_evaluation(
                base_url="http://mock",
                input_file=input_csv,
                output_file=output_csv,
                summary_file=summary_json,
                resume=True,
                retry_errors=True,
            )
            assert eval_calls == 1

            # Another retry-errors run finds no failed records to re-run
            await run_evaluation(
                base_url="http://mock",
                input_file=input_csv,
                output_file=output_csv,
                summary_file=summary_json,
                resume=True,
                retry_errors=True,
            )
            assert eval_calls == 1
        finally:
            rge.evaluate_single_example = orig_evaluate


@pytest.mark.asyncio
async def test_multiple_retries_single_row_per_example():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = Path(tmpdir) / "golden.csv"
        output_csv = Path(tmpdir) / "results.csv"
        summary_json = Path(tmpdir) / "summary.json"

        with open(input_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "example_id", "customer_text", "expected_intent", "expected_escalate",
                "expected_escalation_reason", "expected_reply_points", "language_final", "difficulty_final", "sampling_bucket"
            ])
            writer.writeheader()
            writer.writerow({
                "example_id": "G001", "customer_text": "q1", "expected_intent": "ACCOUNT_ACCESS",
                "expected_escalate": "FALSE", "expected_escalation_reason": "none",
                "expected_reply_points": "reply", "language_final": "en", "difficulty_final": "easy", "sampling_bucket": "b1"
            })

        # Initial failure
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            writer.writerow({
                "example_id": "G001", "customer_text": "q1", "expected_intent": "ACCOUNT_ACCESS",
                "predicted_intent": "", "intent_correct": "False", "expected_escalate": "False",
                "predicted_escalate": "False", "escalation_correct": "False", "expected_escalation_reason": "",
                "predicted_escalation_reason": "", "expected_reply_points": "", "generated_reply": "",
                "retrieved_context_available": "False", "retrieved_context_count": "0", "ttft_ms": "0",
                "total_latency_ms": "0", "difficulty": "easy", "language": "en", "sampling_bucket": "b1",
                "request_status": "FAILED", "error_type": "TIMEOUT", "error_message": "Timeout 1"
            })

        retry_attempt = 1

        async def mock_evaluate_single(client, base_url, row, timeout_sec, **kwargs):
            nonlocal retry_attempt
            if retry_attempt == 1:
                retry_attempt += 1
                return {
                    "example_id": row["example_id"], "customer_text": row["customer_text"],
                    "expected_intent": row["expected_intent"], "predicted_intent": "",
                    "intent_correct": False, "expected_escalate": False, "predicted_escalate": False,
                    "escalation_correct": False, "expected_escalation_reason": "", "predicted_escalation_reason": "",
                    "expected_reply_points": "", "generated_reply": "", "retrieved_context_available": False,
                    "retrieved_context_count": 0, "ttft_ms": 0.0, "total_latency_ms": 0.0,
                    "difficulty": "easy", "language": "en", "sampling_bucket": "b1",
                    "request_status": "FAILED", "error_type": "HTTP_500", "error_message": "Server Error"
                }
            else:
                return {
                    "example_id": row["example_id"], "customer_text": row["customer_text"],
                    "expected_intent": row["expected_intent"], "predicted_intent": "ACCOUNT_ACCESS",
                    "intent_correct": True, "expected_escalate": False, "predicted_escalate": False,
                    "escalation_correct": True, "expected_escalation_reason": "", "predicted_escalation_reason": "",
                    "expected_reply_points": "", "generated_reply": "success reply", "retrieved_context_available": True,
                    "retrieved_context_count": 1, "ttft_ms": 4000.0, "total_latency_ms": 5000.0,
                    "difficulty": "easy", "language": "en", "sampling_bucket": "b1",
                    "request_status": "SUCCESS", "error_type": "", "error_message": ""
                }

        import scripts.evaluation.run_golden_eval as rge
        orig_evaluate = rge.evaluate_single_example
        rge.evaluate_single_example = mock_evaluate_single

        try:
            # Retry 1: still fails
            await run_evaluation(
                base_url="http://mock",
                input_file=input_csv,
                output_file=output_csv,
                summary_file=summary_json,
                resume=True,
                retry_errors=True,
            )
            with open(output_csv, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 1
            assert rows[0]["request_status"] == "FAILED"
            assert rows[0]["error_type"] == "HTTP_500"

            # Retry 2: succeeds
            await run_evaluation(
                base_url="http://mock",
                input_file=input_csv,
                output_file=output_csv,
                summary_file=summary_json,
                resume=True,
                retry_errors=True,
            )
            with open(output_csv, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 1
            assert rows[0]["request_status"] == "SUCCESS"
            assert rows[0]["generated_reply"] == "success reply"
        finally:
            rge.evaluate_single_example = orig_evaluate


def test_save_results_atomically_preserves_golden_order():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_csv = Path(tmpdir) / "results.csv"
        results_by_id = {
            "G003": {col: f"val_G003_{col}" for col in RESULT_COLUMNS},
            "G001": {col: f"val_G001_{col}" for col in RESULT_COLUMNS},
            "G002": {col: f"val_G002_{col}" for col in RESULT_COLUMNS},
        }
        for eid in results_by_id:
            results_by_id[eid]["example_id"] = eid

        ordered_ids = ["G001", "G002", "G003"]
        save_results_atomically(output_csv, results_by_id, ordered_example_ids=ordered_ids)

        with open(output_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 3
        assert [r["example_id"] for r in rows] == ["G001", "G002", "G003"]


@pytest.mark.asyncio
async def test_fresh_run_never_produces_duplicate_or_400_rows():
    """Test 1: Fresh 200-example run can never produce 400 rows when an older 200-row file exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = Path(tmpdir) / "golden.csv"
        output_csv = Path(tmpdir) / "results.csv"
        summary_json = Path(tmpdir) / "summary.json"

        # Create 200 golden examples
        with open(input_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "example_id", "customer_text", "expected_intent", "expected_escalate",
                "expected_escalation_reason", "expected_reply_points", "language_final", "difficulty_final", "sampling_bucket"
            ])
            writer.writeheader()
            for i in range(1, 201):
                eid = f"G{i:03d}"
                writer.writerow({
                    "example_id": eid, "customer_text": f"text {eid}", "expected_intent": "ACCOUNT_ACCESS",
                    "expected_escalate": "FALSE", "expected_escalation_reason": "none",
                    "expected_reply_points": "reply", "language_final": "en", "difficulty_final": "easy", "sampling_bucket": "b1"
                })

        # Pre-populate output CSV with 200 OLD rows
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            for i in range(1, 201):
                eid = f"G{i:03d}"
                writer.writerow({
                    "example_id": eid, "customer_text": f"old text {eid}", "expected_intent": "ACCOUNT_ACCESS",
                    "predicted_intent": "OLD_INTENT", "intent_correct": "False", "expected_escalate": "False",
                    "predicted_escalate": "False", "escalation_correct": "True", "request_status": "SUCCESS",
                })

        async def mock_evaluate_single(client, base_url, row, timeout_sec, run_id="", golden_set_hash=""):
            return {
                "run_id": run_id, "example_id": row["example_id"], "customer_text": row["customer_text"],
                "expected_intent": row["expected_intent"], "predicted_intent": "ACCOUNT_ACCESS",
                "intent_correct": True, "expected_escalate": False, "predicted_escalate": False,
                "escalation_correct": True, "expected_escalation_reason": "", "predicted_escalation_reason": "",
                "expected_reply_points": "", "generated_reply": "fresh reply", "retrieved_context_available": True,
                "retrieved_context_count": 1, "ttft_ms": 100.0, "total_latency_ms": 200.0,
                "difficulty": "easy", "language": "en", "sampling_bucket": "b1",
                "request_status": "SUCCESS", "error_category": "NONE", "error_type": "", "error_message": "",
                "timestamp": "2026-09-13T00:00:00Z", "golden_set_hash": golden_set_hash, "base_url": base_url,
            }

        import scripts.evaluation.run_golden_eval as rge
        orig_evaluate = rge.evaluate_single_example
        rge.evaluate_single_example = mock_evaluate_single

        try:
            # Fresh run (resume=False)
            await run_evaluation(
                base_url="http://mock",
                input_file=input_csv,
                output_file=output_csv,
                summary_file=summary_json,
                concurrency=5,
                resume=False,
            )

            with open(output_csv, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            assert len(rows) == 200, f"Expected exactly 200 rows, got {len(rows)}"
            assert rows[0]["generated_reply"] == "fresh reply"
        finally:
            rge.evaluate_single_example = orig_evaluate


@pytest.mark.asyncio
async def test_changed_golden_labels_update_expected_fields():
    """Test 4: Changed golden labels update expected fields on startup without duplicating rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = Path(tmpdir) / "golden.csv"
        output_csv = Path(tmpdir) / "results.csv"
        summary_json = Path(tmpdir) / "summary.json"

        # Golden set with updated expected_intent for G001
        with open(input_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "example_id", "customer_text", "expected_intent", "expected_escalate",
                "expected_escalation_reason", "expected_reply_points", "language_final", "difficulty_final", "sampling_bucket"
            ])
            writer.writeheader()
            writer.writerow({
                "example_id": "G001", "customer_text": "q1", "expected_intent": "ORDER_STATUS",  # Changed!
                "expected_escalate": "TRUE", "expected_escalation_reason": "none",
                "expected_reply_points": "reply", "language_final": "en", "difficulty_final": "easy", "sampling_bucket": "b1"
            })

        # Pre-populate output CSV with old expected_intent
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            writer.writerow({
                "example_id": "G001", "customer_text": "q1", "expected_intent": "ACCOUNT_ACCESS",
                "predicted_intent": "ORDER_STATUS", "intent_correct": "False", "expected_escalate": "False",
                "predicted_escalate": "True", "escalation_correct": "False", "request_status": "SUCCESS",
            })

        # Run with --resume
        await run_evaluation(
            base_url="http://mock",
            input_file=input_csv,
            output_file=output_csv,
            summary_file=summary_json,
            resume=True,
        )

        with open(output_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1
        assert rows[0]["expected_intent"] == "ORDER_STATUS"
        assert rows[0]["intent_correct"] == "True"
        assert rows[0]["expected_escalate"] == "True"
        assert rows[0]["escalation_correct"] == "True"


@pytest.mark.asyncio
async def test_old_ids_absent_from_current_golden_file_are_dropped():
    """Test 5: Old IDs absent from current golden file are dropped on startup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = Path(tmpdir) / "golden.csv"
        output_csv = Path(tmpdir) / "results.csv"
        summary_json = Path(tmpdir) / "summary.json"

        # Current golden file only has G001 and G002
        with open(input_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "example_id", "customer_text", "expected_intent", "expected_escalate",
                "expected_escalation_reason", "expected_reply_points", "language_final", "difficulty_final", "sampling_bucket"
            ])
            writer.writeheader()
            for eid in ["G001", "G002"]:
                writer.writerow({
                    "example_id": eid, "customer_text": f"q {eid}", "expected_intent": "ACCOUNT_ACCESS",
                    "expected_escalate": "FALSE", "expected_escalation_reason": "none",
                    "expected_reply_points": "reply", "language_final": "en", "difficulty_final": "easy", "sampling_bucket": "b1"
                })

        # Pre-populate output CSV with G001, G002, and stale G999
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            for eid in ["G001", "G002", "G999"]:
                writer.writerow({
                    "example_id": eid, "customer_text": f"q {eid}", "expected_intent": "ACCOUNT_ACCESS",
                    "predicted_intent": "ACCOUNT_ACCESS", "intent_correct": "True", "request_status": "SUCCESS",
                })

        await run_evaluation(
            base_url="http://mock",
            input_file=input_csv,
            output_file=output_csv,
            summary_file=summary_json,
            resume=True,
        )

        with open(output_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2
        assert [r["example_id"] for r in rows] == ["G001", "G002"]


@pytest.mark.asyncio
async def test_two_consecutive_endpoint_404s_stop_scheduling_requests():
    """Test 6: Two consecutive endpoint 404s trigger the infrastructure circuit breaker and stop."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = Path(tmpdir) / "golden.csv"
        output_csv = Path(tmpdir) / "results.csv"
        summary_json = Path(tmpdir) / "summary.json"

        # 10 golden examples
        with open(input_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "example_id", "customer_text", "expected_intent", "expected_escalate",
                "expected_escalation_reason", "expected_reply_points", "language_final", "difficulty_final", "sampling_bucket"
            ])
            writer.writeheader()
            for i in range(1, 11):
                eid = f"G{i:03d}"
                writer.writerow({
                    "example_id": eid, "customer_text": f"q {eid}", "expected_intent": "ACCOUNT_ACCESS",
                    "expected_escalate": "FALSE", "expected_escalation_reason": "none",
                    "expected_reply_points": "reply", "language_final": "en", "difficulty_final": "easy", "sampling_bucket": "b1"
                })

        evaluated_ids = []

        async def mock_evaluate_single(client, base_url, row, timeout_sec, run_id="", golden_set_hash=""):
            evaluated_ids.append(row["example_id"])
            return {
                "run_id": run_id, "example_id": row["example_id"], "customer_text": row["customer_text"],
                "expected_intent": row["expected_intent"], "predicted_intent": "",
                "intent_correct": False, "expected_escalate": False, "predicted_escalate": False,
                "escalation_correct": False, "expected_escalation_reason": "", "predicted_escalation_reason": "",
                "expected_reply_points": "", "generated_reply": "", "retrieved_context_available": False,
                "retrieved_context_count": 0, "ttft_ms": 0.0, "total_latency_ms": 0.0,
                "difficulty": "easy", "language": "en", "sampling_bucket": "b1",
                "request_status": "FAILED", "error_category": "INFRASTRUCTURE",
                "error_type": "ENDPOINT_NOT_FOUND", "error_message": "404 Not Found",
                "timestamp": "2026-09-13T00:00:00Z", "golden_set_hash": golden_set_hash, "base_url": base_url,
            }

        import scripts.evaluation.run_golden_eval as rge
        orig_evaluate = rge.evaluate_single_example
        rge.evaluate_single_example = mock_evaluate_single

        try:
            await run_evaluation(
                base_url="http://mock",
                input_file=input_csv,
                output_file=output_csv,
                summary_file=summary_json,
                concurrency=1,
                consecutive_infra_threshold=2,
            )

            # Circuit breaker should have stopped after exactly 2 consecutive 404s
            assert len(evaluated_ids) == 2
            assert evaluated_ids == ["G001", "G002"]

            with open(output_csv, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 2
        finally:
            rge.evaluate_single_example = orig_evaluate


def test_404_is_infrastructure_failure_not_model_performance():
    """Test 7: 404 is categorized as infrastructure failure and excluded from model performance."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerow({
            "example_id": "G001", "customer_text": "q1", "expected_intent": "ACCOUNT_ACCESS",
            "predicted_intent": "", "intent_correct": "False", "expected_escalate": "False",
            "predicted_escalate": "False", "escalation_correct": "False", "request_status": "FAILED",
            "error_category": "INFRASTRUCTURE", "error_type": "ENDPOINT_NOT_FOUND",
            "error_message": "404 Not Found on /api/v1/chat/stream",
        })
        tmp_csv = f.name

    try:
        metrics = calculate_metrics_from_csv(tmp_csv, max_expected_examples=1)
        assert metrics["n_total"] == 1
        assert metrics["n_success"] == 0
        assert metrics["n_failed"] == 1
        assert metrics["infrastructure_failures"] == 1
        assert metrics["model_successful_examples"] == 0
        assert metrics["intent"]["accuracy"] == 0.0
    finally:
        Path(tmp_csv).unlink(missing_ok=True)


def test_metrics_reject_duplicate_example_ids():
    """Test 8: Metrics calculation raises ValueError on duplicate example_ids."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerow({"example_id": "G001", "request_status": "SUCCESS"})
        writer.writerow({"example_id": "G001", "request_status": "FAILED"})
        tmp_csv = f.name

    try:
        with pytest.raises(ValueError, match="duplicate example_ids"):
            calculate_metrics_from_csv(tmp_csv, strict_no_duplicates=True)
    finally:
        Path(tmp_csv).unlink(missing_ok=True)


def test_metrics_reject_result_count_greater_than_golden_set():
    """Test 9: Metrics calculation raises ValueError if row count exceeds max allowed golden examples."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for i in range(1, 5):
            writer.writerow({"example_id": f"G00{i}", "request_status": "SUCCESS"})
        tmp_csv = f.name

    try:
        # Max allowed is 3, but 4 rows provided
        with pytest.raises(ValueError, match="exceeds max allowed"):
            calculate_metrics_from_csv(tmp_csv, max_expected_examples=3)
    finally:
        Path(tmp_csv).unlink(missing_ok=True)


def test_atomic_rewrite_preserves_results_after_interruption():
    """Test 10: Atomic write ensures complete, valid CSV snapshot on disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_csv = Path(tmpdir) / "results.csv"
        results = {
            "G001": {col: f"val_1_{col}" for col in RESULT_COLUMNS},
            "G002": {col: f"val_2_{col}" for col in RESULT_COLUMNS},
        }
        results["G001"]["example_id"] = "G001"
        results["G002"]["example_id"] = "G002"

        save_results_atomically(output_csv, results, ordered_example_ids=["G001", "G002"])

        loaded = load_existing_results(output_csv)
        assert len(loaded) == 2
        assert "G001" in loaded and "G002" in loaded


def test_eight_intent_taxonomy_with_independent_escalation_metrics():
    """Verify metrics calculation operates on the 8 canonical issue intents without ESCALATION as an intent."""
    canonical_8_intents = [
        "ACCOUNT_ACCESS",
        "ACCOUNT_SUPPORT",
        "CUSTOMER_SERVICE_CONTACT",
        "DELIVERY_DELAY",
        "ORDER_STATUS",
        "PACKAGE_NOT_RECEIVED",
        "PRODUCT_ISSUE",
        "REFUND_PENDING",
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for idx, intent in enumerate(canonical_8_intents, 1):
            writer.writerow({
                "example_id": f"G{idx:03d}",
                "expected_intent": intent,
                "predicted_intent": intent,
                "intent_correct": "True",
                "expected_escalate": "True" if idx % 2 == 0 else "False",
                "predicted_escalate": "True" if idx % 2 == 0 else "False",
                "escalation_correct": "True",
                "retrieved_context_available": "True",
                "retrieved_context_count": "1",
                "ttft_ms": "1000",
                "total_latency_ms": "2000",
                "request_status": "SUCCESS",
                "error_category": "NONE",
            })
        tmp_csv = f.name

    try:
        metrics = calculate_metrics_from_csv(tmp_csv, max_expected_examples=10)
        assert metrics["n_total"] == 8
        assert metrics["n_success"] == 8
        assert metrics["intent"]["accuracy"] == 1.0
        assert metrics["escalation"]["accuracy"] == 1.0
        assert metrics["escalation"]["f1"] == 1.0

        # Verify all 8 intents exist in per_intent metrics and none is ESCALATION
        per_intent = metrics["intent"]["per_intent"]
        assert sorted(list(per_intent.keys())) == sorted(canonical_8_intents)
        assert "ESCALATION" not in per_intent
    finally:
        Path(tmp_csv).unlink(missing_ok=True)



