# tests/test_nlp_utils.py

from unittest.mock import MagicMock

import pytest

from src.nlp_utils import get_keywords, load_nlp


def test_load_nlp_success():
    """
    Test that loading the default 'en_core_web_sm' spaCy model succeeds (if installed).
    """
    try:
        nlp = load_nlp("en_core_web_sm")
        assert nlp is not None, "Failed to load spaCy model en_core_web_sm"
    except OSError:
        pytest.skip("spaCy model en_core_web_sm is not installed. Skipping test.")


def test_get_keywords_basic():
    """
    Test the get_keywords function with a straightforward sentence.
    """
    # Mock the nlp object to return a simple tokenization result
    # For a real test, you might actually load a spaCy model;
    # but here we show how to mock to keep tests fast & isolated.
    mock_nlp = MagicMock()

    # Create a mock doc with token objects
    # Each token can have pos_, text, etc.
    Token = MagicMock
    tokens = [
        Token(text="galaxy", pos_="NOUN"),
        Token(text="evolution", pos_="NOUN"),
        Token(text="is", pos_="VERB"),
        Token(text="interesting", pos_="ADJ"),
        Token(text="!", pos_="PUNCT"),
    ]

    # Make doc iteration yield tokens
    mock_nlp.return_value = tokens

    text = "Galaxy evolution is interesting!"
    keywords = get_keywords(text, mock_nlp)

    # We expect 'galaxy', 'evolution', and 'interesting'
    # 'is' is not in pos_tags, '!' is punctuation
    assert keywords == ["galaxy", "evolution", "interesting"], f"Got {keywords}"


def test_get_keywords_empty():
    """
    Test that get_keywords returns an empty list for empty or None text.
    """
    # Actual spaCy model is not needed if text is empty.
    nlp = MagicMock()

    assert get_keywords("", nlp) == [], "Should return empty list for empty string"
    assert get_keywords(None, nlp) == [], "Should return empty list for None text"


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "Dark Matter is crucial in Cosmology!",
            ["dark", "matter", "crucial", "cosmology"],
        ),
        (
            "I love python packaging, but it's tricky sometimes.",
            ["python", "packaging", "tricky"],
        ),
        ("", []),
    ],
)
def test_get_keywords_integration(text, expected):
    """
    An integration-style test using a real spaCy model (if available).
    Only runs if 'en_core_web_sm' is installed.
    """
    try:
        real_nlp = load_nlp("en_core_web_sm")
    except OSError:
        pytest.skip(
            "spaCy model en_core_web_sm is not installed. Skipping integration test."
        )

    result = get_keywords(text, real_nlp)
    # Because we do lowercasing, we also lower the 'expected' list
    assert sorted(result) == sorted(expected), f"For input: {text}, got {result}"
