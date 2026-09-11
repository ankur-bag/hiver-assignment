"""
ML Pipeline Configuration.
Defines model paths, embedding configurations, confidence thresholds,
and canonical intent taxonomy.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import os


# Base paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


@dataclass(frozen=True)
class MLConfig:
    """Configuration settings for the ML Intent Inference Service."""

    # Sentence Transformer embedding model identifier
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Embedding dimension for MiniLM-L12-v2
    EMBEDDING_DIMENSION: int = 384

    # Classifier model artifact paths
    CLASSIFIER_PATH: Path = field(
        default_factory=lambda: MODELS_DIR / "embedding_intent_classifier.pkl"
    )
    WEIGHTS_PATH: Path = field(
        default_factory=lambda: MODELS_DIR / "classifier_weights.npz"
    )

    # Label mapping path
    LABEL_MAP_PATH: Path = field(
        default_factory=lambda: MODELS_DIR / "label_map.json"
    )

    # Metadata file path
    METADATA_PATH: Path = field(
        default_factory=lambda: MODELS_DIR / "model_metadata.json"
    )

    # Minimum confidence required before triggering fallback / escalation flag
    CONFIDENCE_THRESHOLD: float = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.50"))

    # Canonical list of 9 supported intents
    SUPPORTED_INTENTS: List[str] = field(
        default_factory=lambda: [
            "ACCOUNT_ACCESS",
            "ACCOUNT_SUPPORT",
            "CUSTOMER_SERVICE_CONTACT",
            "DELIVERY_DELAY",
            "ESCALATION",
            "ORDER_STATUS",
            "PACKAGE_NOT_RECEIVED",
            "PRODUCT_ISSUE",
            "REFUND_PENDING",
        ]
    )

    # Fallback intent if confidence is below threshold or query is ambiguous
    FALLBACK_INTENT: str = "CUSTOMER_SERVICE_CONTACT"


# Default singleton instance
config = MLConfig()
