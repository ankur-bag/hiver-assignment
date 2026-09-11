"""
Intent Inference Engine.
Provides production-grade inference, confidence calculation, batch prediction,
fallback handling, and input validation using lightweight NumPy matrix operations.
"""

import json
import logging
from threading import Lock
from typing import Any, Dict, List, Optional
import numpy as np

from ml.config import MLConfig, config as default_config
from ml.embeddings.encoder import EmbeddingEncoder, get_embedding_encoder
from ml.preprocessing import clean_text, validate_and_clean_query, InputValidationError

logger = logging.getLogger(__name__)


class IntentClassifierService:
    """
    Production Intent Inference Service.
    Loads the trained model weights and connects with ONNX normalized embeddings
    to generate intent classifications and confidence scores using pure NumPy.
    """

    def __init__(
        self,
        config: Optional[MLConfig] = None,
        encoder: Optional[EmbeddingEncoder] = None
    ):
        self.config = config or default_config
        self.encoder = encoder or get_embedding_encoder()
        self._load_model()

    def _load_model(self) -> None:
        """Loads and verifies the serialized intent classification weights artifact."""
        weights_path = self.config.WEIGHTS_PATH
        pkl_path = self.config.CLASSIFIER_PATH

        if weights_path.exists():
            logger.info(f"Loading lightweight classifier weights from {weights_path}...")
            weights = np.load(weights_path, allow_pickle=True)
            self.classes_ = list(weights["classes"].astype(str))
            self.coef_ = weights["coef"].astype(np.float32)
            self.intercept_ = weights["intercept"].astype(np.float32)
            self._uses_sklearn = False
            logger.info(f"NumPy intent classifier loaded successfully with classes: {self.classes_}")
        elif pkl_path.exists():
            logger.info(f"Loading legacy intent classifier from {pkl_path}...")
            import joblib
            self.model = joblib.load(pkl_path)
            self.classes_ = list(self.model.classes_)
            self.coef_ = self.model.coef_.astype(np.float32)
            self.intercept_ = self.model.intercept_.astype(np.float32)
            self._uses_sklearn = False
            logger.info(f"Intent classifier loaded successfully with classes: {self.classes_}")
        else:
            raise FileNotFoundError(
                f"No classifier artifacts found at {weights_path} or {pkl_path}."
            )

    def _predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Pure NumPy Softmax classification:
        logits = embeddings @ coef_.T + intercept_
        probabilities = softmax(logits)
        """
        logits = np.dot(embeddings, self.coef_.T) + self.intercept_
        # Numerical stability shift
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return probabilities

    def predict(
        self,
        text: str,
        include_metadata: bool = False
    ) -> Dict[str, Any]:
        """
        Classifies a single customer query text.
        
        Args:
            text: Raw input string from customer.
            include_metadata: If True, attaches detailed scores, cleaned text,
                             and fallback flag.
                             
        Returns:
            Dict containing:
            {
                "intent": "PACKAGE_NOT_RECEIVED",
                "confidence": 0.97
            }
            Plus additional metadata if include_metadata is True.
        """
        try:
            cleaned, is_valid = validate_and_clean_query(text)
        except InputValidationError as e:
            logger.warning(f"Input validation error: {e}")
            if include_metadata:
                return {
                    "intent": self.config.FALLBACK_INTENT,
                    "confidence": 0.0,
                    "is_fallback": True,
                    "error": str(e)
                }
            return {
                "intent": self.config.FALLBACK_INTENT,
                "confidence": 0.0
            }

        # Handle empty/whitespace input
        if not is_valid or not cleaned:
            if include_metadata:
                return {
                    "intent": self.config.FALLBACK_INTENT,
                    "confidence": 0.0,
                    "is_fallback": True,
                    "error": "Empty or whitespace-only input."
                }
            return {
                "intent": self.config.FALLBACK_INTENT,
                "confidence": 0.0
            }

        # Generate normalized embedding vector (384,)
        embedding = self.encoder.encode(cleaned, show_progress_bar=False)
        embedding_2d = np.expand_dims(embedding, axis=0)  # Shape (1, 384)

        # Compute class probabilities via NumPy softmax
        probabilities = self._predict_proba(embedding_2d)[0]
        best_idx = int(np.argmax(probabilities))
        predicted_intent = str(self.classes_[best_idx])
        raw_confidence = float(probabilities[best_idx])
        rounded_confidence = round(raw_confidence, 2)

        # Check confidence threshold for human escalation / fallback
        is_fallback = raw_confidence < self.config.CONFIDENCE_THRESHOLD

        result: Dict[str, Any] = {
            "intent": predicted_intent,
            "confidence": rounded_confidence
        }

        if include_metadata:
            scores_dict = {
                cls_name: round(float(prob), 4)
                for cls_name, prob in zip(self.classes_, probabilities)
            }
            result.update({
                "raw_confidence": round(raw_confidence, 4),
                "is_fallback": is_fallback,
                "cleaned_text": cleaned,
                "all_scores": scores_dict
            })

        return result

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        include_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Classifies a batch of customer queries efficiently.
        
        Args:
            texts: List of raw strings.
            batch_size: Batch size for embedding generation.
            include_metadata: Whether to return full score metadata.
            
        Returns:
            List of prediction dictionaries.
        """
        if not texts:
            return []

        cleaned_list = []
        valid_indices = []
        invalid_results: Dict[int, Dict[str, Any]] = {}

        for i, t in enumerate(texts):
            try:
                cleaned, is_valid = validate_and_clean_query(t)
                if is_valid and cleaned:
                    valid_indices.append(i)
                    cleaned_list.append(cleaned)
                else:
                    invalid_results[i] = {
                        "intent": self.config.FALLBACK_INTENT,
                        "confidence": 0.0,
                        **({"is_fallback": True, "error": "Empty or whitespace-only input"} if include_metadata else {})
                    }
            except InputValidationError as e:
                invalid_results[i] = {
                    "intent": self.config.FALLBACK_INTENT,
                    "confidence": 0.0,
                    **({"is_fallback": True, "error": str(e)} if include_metadata else {})
                }

        results: List[Optional[Dict[str, Any]]] = [None] * len(texts)
        for idx, res in invalid_results.items():
            results[idx] = res

        if cleaned_list:
            embeddings = self.encoder.encode(cleaned_list, batch_size=batch_size, show_progress_bar=False)
            proba_batch = self._predict_proba(embeddings)

            for original_idx, probs in zip(valid_indices, proba_batch):
                best_idx = int(np.argmax(probs))
                predicted_intent = str(self.classes_[best_idx])
                raw_conf = float(probs[best_idx])
                rounded_conf = round(raw_conf, 2)
                is_fallback = raw_conf < self.config.CONFIDENCE_THRESHOLD

                res_dict = {
                    "intent": predicted_intent,
                    "confidence": rounded_conf
                }

                if include_metadata:
                    scores_dict = {
                        cls_name: round(float(p), 4)
                        for cls_name, p in zip(self.classes_, probs)
                    }
                    res_dict.update({
                        "raw_confidence": round(raw_conf, 4),
                        "is_fallback": is_fallback,
                        "cleaned_text": cleaned_list[valid_indices.index(original_idx)],
                        "all_scores": scores_dict
                    })

                results[original_idx] = res_dict

        return results  # type: ignore


# Global singleton instance and thread lock
_classifier_service: Optional[IntentClassifierService] = None
_service_lock = Lock()


def get_intent_classifier() -> IntentClassifierService:
    """Returns thread-safe singleton instance of IntentClassifierService."""
    global _classifier_service
    if _classifier_service is None:
        with _service_lock:
            if _classifier_service is None:
                _classifier_service = IntentClassifierService()
    return _classifier_service
