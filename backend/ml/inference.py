import json
import logging
from threading import Lock
from typing import Any, Dict, List, Optional, Union
import numpy as np

from ml.config import MLConfig, config as default_config
from ml.preprocessing import clean_text, validate_and_clean_query, InputValidationError

logger = logging.getLogger(__name__)


class IntentClassifierService:
    """
    Production Intent Inference Service using Gemini Embedding 2 and NumPy.
    Loads trained Logistic Regression weights (classes, coef, intercept) and executes
    fast, pure NumPy dot-product and softmax inference (<0.5ms).
    """

    def __init__(
        self,
        config: Optional[MLConfig] = None,
        embedding_service: Optional[Any] = None
    ):
        self.config = config or default_config
        self._embedding_service = embedding_service
        self._load_model()

    @property
    def embedding_service(self):
        """Lazy-loads GeminiEmbeddingService to prevent startup circular import."""
        if self._embedding_service is None:
            try:
                from services.embedding_service import get_embedding_service
                self._embedding_service = get_embedding_service()
            except ImportError:
                # Fallback for isolated ML evaluations
                from ml.embeddings.encoder import get_embedding_encoder
                self._embedding_service = get_embedding_encoder()
        return self._embedding_service

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
            logger.info(f"NumPy intent classifier loaded successfully with {len(self.classes_)} classes.")
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
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)

        logits = np.dot(embeddings, self.coef_.T) + self.intercept_
        # Numerical stability shift
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return probabilities

    def predict_with_vector(
        self,
        query_vector: Union[np.ndarray, List[float]],
        include_metadata: bool = False,
        cleaned_text: str = ""
    ) -> Dict[str, Any]:
        """
        Classifies a customer intent using an already computed embedding vector.
        Eliminates duplicate remote embedding API calls during production chat inference.
        """
        vec = np.asarray(query_vector, dtype=np.float32)
        if vec.ndim == 1:
            vec_2d = np.expand_dims(vec, axis=0)
        else:
            vec_2d = vec

        probabilities = self._predict_proba(vec_2d)[0]
        best_idx = int(np.argmax(probabilities))
        predicted_intent = str(self.classes_[best_idx])
        raw_confidence = float(probabilities[best_idx])
        rounded_confidence = round(raw_confidence, 2)

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
                "cleaned_text": cleaned_text,
                "all_scores": scores_dict
            })

        return result

    def predict(
        self,
        text: str,
        include_metadata: bool = False
    ) -> Dict[str, Any]:
        """
        Classifies a single customer query text via Gemini embedding + NumPy Softmax.
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

        # Generate normalized embedding vector (384,) via Gemini Embedding Service
        if hasattr(self.embedding_service, "embed_query"):
            embedding = self.embedding_service.embed_query(cleaned)
        elif hasattr(self.embedding_service, "encode"):
            embedding = self.embedding_service.encode(cleaned, show_progress_bar=False)
        else:
            raise AttributeError("Embedding service does not provide embed_query or encode.")

        return self.predict_with_vector(
            query_vector=embedding,
            include_metadata=include_metadata,
            cleaned_text=cleaned
        )

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        include_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Classifies a batch of customer queries efficiently.
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
            if hasattr(self.embedding_service, "embed_documents"):
                embeddings = self.embedding_service.embed_documents(cleaned_list, batch_size=batch_size)
            elif hasattr(self.embedding_service, "encode"):
                embeddings = self.embedding_service.encode(cleaned_list, batch_size=batch_size, show_progress_bar=False)
            else:
                embeddings = np.array([self.embedding_service.embed_query(c) for c in cleaned_list], dtype=np.float32)

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
