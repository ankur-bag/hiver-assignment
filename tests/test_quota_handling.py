from unittest.mock import MagicMock
import pytest

from services.gemini_service import GeminiFileSearchService, ProviderQuotaError


def test_daily_quota_fails_without_retry():
    service = GeminiFileSearchService("key", "fileSearchStores/test", "model")
    operation = MagicMock(side_effect=Exception("429 RESOURCE_EXHAUSTED daily quota"))
    with pytest.raises(ProviderQuotaError):
        service._call_with_retry(operation)
    assert operation.call_count == 1
