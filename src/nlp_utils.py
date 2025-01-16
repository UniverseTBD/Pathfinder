# src/nlp_utils.py

from string import punctuation
from typing import List

import nltk
import spacy
from nltk.corpus import stopwords
from spacy.language import Language


def load_nlp(model_name: str = "en_core_web_sm") -> Language:
    """
    Load a spaCy model and ensure supporting resources (e.g. NLTK stopwords) are present.

    This function attempts to load the specified spaCy model. If the model is not found,
    it prints installation instructions and raises an exception. Additionally, it verifies
    that NLTK stopwords are available, downloading them if necessary.

    Args:
        model_name (str): The name of the spaCy model to load. Defaults to "en_core_web_sm".

    Returns:
        Language: The loaded spaCy Language model.

    Raises:
        OSError: If the specified spaCy model is not installed.
        LookupError: If NLTK stopwords are unavailable and fail to download.
    """
    # Attempt to load the spaCy model
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"Model '{model_name}' not found. Install it via:")
        print(f"    python -m spacy download {model_name}")
        raise

    # Check for NLTK stopwords
    try:
        stopwords.words("english")
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download("stopwords", quiet=True)

    # If you have any additional pipeline components, add them here
    # e.g. nlp.add_pipe("textrank")

    return nlp


def get_keywords(text: str, nlp: Language) -> List[str]:
    """
    Extract keywords from the given text using spaCy for linguistic processing.

    This function processes the input text using a spaCy Language model to identify
    potential keywords. Keywords are extracted based on:
      1. Being a PROPN (proper noun), ADJ (adjective), or NOUN part-of-speech.
      2. Not being an NLTK stopword.
      3. Not being punctuation.

    Args:
        text (str): The input text from which to extract keywords.
        nlp (Language): A loaded spaCy Language model for text processing.

    Returns:
        List[str]: A list of extracted keywords (in lowercase).
    """
    # Handle edge case: empty or None text
    if not text:
        return []

    # The part-of-speech tags we'll consider as potential keywords
    target_pos_tags = {"PROPN", "ADJ", "NOUN"}

    # Process the text (lowercased) via spaCy
    doc = nlp(text.lower())

    # Filter out punctuation, stopwords, and non-target POS tokens
    keywords = [
        token.text
        for token in doc
        if (
            token.pos_ in target_pos_tags
            and token.text not in punctuation
            and token.text not in nlp.Defaults.stop_words
        )
    ]

    return keywords
