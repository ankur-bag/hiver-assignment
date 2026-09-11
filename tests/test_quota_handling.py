from unittest.mock import MagicMock
import pytest

from services.gemini_service import GeminiFileSearchService, ProviderQuotaError


def test_daily_quota_fails_without_retry():
    service = GeminiFileSearchService("key", "fileSearchStores/test", "model")
    operation = MagicMock(side_effect=Exception("429 RESOURCE_EXHAUSTED daily quota"))
    with pytest.raises(ProviderQuotaError):
        service._call_with_retry(operation)
    assert operation.call_count == 1


def test_503_retries_at_most_once():
    from services.gemini_service import ProviderTransientError
    service = GeminiFileSearchService("key", "fileSearchStores/test", "model")
    operation = MagicMock(side_effect=Exception("503 Service Unavailable"))
    with pytest.raises(ProviderTransientError):
        service._call_with_retry(operation)
    assert operation.call_count == 2


def test_400_fails_without_retry():
    from services.gemini_service import ProviderError
    service = GeminiFileSearchService("key", "fileSearchStores/test", "model")
    operation = MagicMock(side_effect=Exception("400 Bad Request"))
    with pytest.raises(ProviderError):
        service._call_with_retry(operation)
    assert operation.call_count == 1
