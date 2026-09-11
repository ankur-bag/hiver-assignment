"""
Pinecone Index Provisioning and Setup Script.
Initializes the Pinecone client, checks for the target serverless index,
provisions it with 384 dimensions and cosine similarity metric,
and verifies index readiness.
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from backend/.env
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / "backend" / ".env"
load_dotenv(ENV_FILE)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "amazon-support-resolutions")
CLOUD = "aws"
REGION = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
DIMENSION = 384
METRIC = "cosine"


def setup_pinecone_index():
    print("=== Pinecone Index Provisioning ===")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY environment variable is missing in backend/.env.")

    print(f"Connecting to Pinecone client...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [idx.name for idx in pc.list_indexes()]
    print(f"Existing indexes in account: {existing_indexes}")

    if INDEX_NAME in existing_indexes:
        print(f"Index '{INDEX_NAME}' already exists.")
        desc = pc.describe_index(INDEX_NAME)
        print(f"Index Status: {desc.status.state} (Ready: {desc.status.ready})")
        print(f"Host: {desc.host}")
        print(f"Dimension: {desc.dimension}, Metric: {desc.metric}")
        if desc.dimension != DIMENSION:
            raise ValueError(
                f"Dimension mismatch! Existing index has {desc.dimension}, but model produces {DIMENSION}."
            )
        return desc

    print(f"Creating serverless index '{INDEX_NAME}'...")
    print(f"  Dimension: {DIMENSION}")
    print(f"  Metric: {METRIC}")
    print(f"  Cloud: {CLOUD}, Region: {REGION}")

    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(
            cloud=CLOUD,
            region=REGION
        )
    )

    print(f"Waiting for index '{INDEX_NAME}' to initialize...")
    while True:
        desc = pc.describe_index(INDEX_NAME)
        if desc.status.ready:
            print(f"Index '{INDEX_NAME}' is now READY!")
            print(f"Host: {desc.host}")
            break
        print("  Index state:", desc.status.state, "... waiting 3 seconds")
        time.sleep(3)

    return desc


if __name__ == "__main__":
    setup_pinecone_index()
