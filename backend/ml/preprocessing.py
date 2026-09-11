"""
Text Preprocessing and Input Validation Module.
Cleans raw customer queries, strips noise, preserves multilingual text,
and validates inputs prior to embedding and classification.
"""

import re
from typing import Tuple


# Regex patterns for noise removal
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HANDLE_PATTERN = re.compile(r"@[A-Za-z0-9_]+", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")

# Maximum character length to prevent denial-of-service / memory overflow
MAX_TEXT_LENGTH = 1000


class InputValidationError(ValueError):
    """Raised when query input fails validation criteria."""
    pass


def clean_text(text: str) -> str:
    """
    Cleans raw customer input:
    - Removes URL links
    - Removes social media handles (e.g. @AmazonHelp)
    - Normalizes extra whitespace while preserving punctuation & unicode
    
    Args:
        text: Raw user query string.
        
    Returns:
        Cleaned, stripped string.
    """
    if not isinstance(text, str):
        return ""
    
    # Strip URLs
    text = URL_PATTERN.sub(" ", text)
    
    # Strip social handles
    text = HANDLE_PATTERN.sub(" ", text)
    
    # Collapse multiple whitespace characters
    text = WHITESPACE_PATTERN.sub(" ", text)
    
    return text.strip()


def validate_and_clean_query(text: str) -> Tuple[str, bool]:
    """
    Validates and cleans incoming customer text query.
    
    Args:
        text: Raw user input text.
        
    Returns:
        Tuple of (cleaned_text, is_valid)
        
    Raises:
        InputValidationError: If input is None or not a string.
    """
    if text is None:
        raise InputValidationError("Input text cannot be None.")
    
    if not isinstance(text, str):
        raise InputValidationError(f"Expected str, received {type(text).__name__}.")
    
    cleaned = clean_text(text)
    
    if len(cleaned) == 0:
        return "", False
    
    # Truncate if exceeds safety limit
    if len(cleaned) > MAX_TEXT_LENGTH:
        cleaned = cleaned[:MAX_TEXT_LENGTH].rstrip()
        
    return cleaned, True
