"""
Interactive Retrieval Testing Script.
Tests two-stage hybrid retrieval against Pinecone vector store:
1. Predicts intent from input query.
2. Performs filtered vector search in 'amazon' namespace.
3. Formats and prints matching historical resolutions and RAG context.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.inference import get_intent_classifier
from backend.services.pinecone_service import get_pinecone_service
from backend.services.context_builder import build_rag_context


def test_retrieval(query: str, top_k: int = 3):
    print("=" * 70)
    print("=== HYBRID SEMANTIC RETRIEVAL TEST ===")
    print("=" * 70)
    print(f"\nUser Query: \"{query}\"")

    # Step 1: Intent Classification
    clf = get_intent_classifier()
    intent_res = clf.predict(query, include_metadata=True)
    predicted_intent = intent_res["intent"]
    confidence = intent_res["confidence"]
    print(f"\n[Stage 1: Intent Classification]")
    print(f"  Predicted Intent: {predicted_intent}")
    print(f"  Confidence Score: {confidence:.2f}")
    print(f"  Fallback Flag:    {intent_res.get('is_fallback', False)}")

    # Step 2: Vector Retrieval in Pinecone
    print(f"\n[Stage 2: Pinecone Vector Retrieval (namespace='amazon', top_k={top_k})]")
    service = get_pinecone_service()
    search_res = service.retrieve_similar_cases(query=query, intent=predicted_intent, top_k=top_k)

    status = search_res.get("retrieval_status")
    results = search_res.get("results", [])
    telemetry = search_res.get("telemetry", {})

    print(f"  Retrieval Status: {status}")
    print(f"  Fallback Used:    {telemetry.get('fallback_used')}")
    print(f"  Query Latency:    {telemetry.get('latency_ms')} ms")
    print(f"  Matches Found:    {len(results)}")

    if not results:
        print("\n  [Notice] No matching vectors found. Ensure upload_pinecone.py has completed.")
        return

    print("\n" + "-" * 70)
    print("TOP RETRIEVED HISTORICAL RESOLUTIONS:")
    print("-" * 70)

    for i, res in enumerate(results, 1):
        print(f"\nResult #{i} | Similarity Score: {res['score']:.4f} | Intent: {res['intent']}")
        print(f"  Historical Query: \"{res['customer_text']}\"")
        print(f"  Amazon Reply:     \"{res['amazon_reply']}\"")

    # Step 3: Structured Context Assembly for Phase 3 Gemini
    print("\n" + "=" * 70)
    print("ASSEMBLED RAG PROMPT CONTEXT (PHASE 3 READY):")
    print("=" * 70)
    rag_context = build_rag_context(results, predicted_intent=predicted_intent)
    print(rag_context)
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test semantic retrieval in Pinecone.")
    parser.add_argument(
        "--query",
        type=str,
        default="My parcel has not arrived yet",
        help="Customer query to test"
    )
    parser.add_argument("--top_k", type=int, default=3, help="Number of results to retrieve")
    args = parser.parse_args()

    test_retrieval(query=args.query, top_k=args.top_k)
