"""
utils/text.py — Text normalisation shared by the API, workers, and Ankush's retrain script.

normalize() must be called on every text string before tokenisation.
If Ankush's retrain.py calls a different normaliser, train/serve text will drift
and the model will see slightly different inputs at inference time.
"""

import re
import unicodedata
from typing import List

# Typographic → ASCII quote mapping applied after NFKC.
# NFKC handles NBSP, full-width chars, etc. but does NOT map curly quotes.
_SMART_QUOTE_MAP = str.maketrans({
    "‘": "'",   # LEFT SINGLE QUOTATION MARK
    "’": "'",   # RIGHT SINGLE QUOTATION MARK
    "‚": ",",   # SINGLE LOW-9 QUOTATION MARK
    "‛": "'",   # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    "“": '"',   # LEFT DOUBLE QUOTATION MARK
    "”": '"',   # RIGHT DOUBLE QUOTATION MARK
    "„": '"',   # DOUBLE LOW-9 QUOTATION MARK
    "‟": '"',   # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    "′": "'",   # PRIME
    "″": '"',   # DOUBLE PRIME
})

# Single newline that is NOT part of a paragraph break (double newline).
# Lookbehind/lookahead ensure we don't consume the \n from \n\n.
_SINGLE_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")

# Runs of horizontal space (but not newlines).
_SPACE_RUN = re.compile(r"[ \t]+")


def normalize(text: str) -> str:
    """
    Normalise legal text before tokenisation.

    Steps (order matters):
      1. NFKC — converts NBSP (U+00A0) to space, full-width chars, etc.
      2. Smart-quote replacement — NFKC does not map typographic quotes.
      3. De-hyphenate line breaks — PDFs often break "hyphen-\\nnation" at word ends.
      4. Unwrap single newlines — sentence-wrapped text → single paragraph;
         paragraph breaks (double newline) are preserved.
      5. Collapse horizontal space runs.
      6. Strip leading/trailing whitespace.

    Args:
        text: Raw text from user input, CSV, or PDF extraction.

    Returns:
        Normalised string ready for the tokeniser.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_SMART_QUOTE_MAP)
    text = text.replace("-\n", "")          # de-hyphenate word-break artifacts
    text = _SINGLE_NEWLINE.sub(" ", text)   # unwrap soft line breaks
    text = _SPACE_RUN.sub(" ", text)        # collapse space/tab runs
    return text.strip()


def count_tokens(text: str) -> int:
    """
    Return the number of tokens the model tokeniser produces for *text*.

    Counts WITHOUT truncation so the caller can detect whether the text
    exceeds MAX_LENGTH and set the ``truncated`` flag accordingly.

    Requires the model to be loaded (model_service.load() must have been called).
    Raises RuntimeError if the tokeniser is not yet available.

    Args:
        text: Already-normalised text (call normalize() first).

    Returns:
        Token count including special tokens ([CLS], [SEP]).
    """
    from services.model_service import model_service  # lazy import — avoids circular dep at startup

    tokenizer = model_service.get_tokenizer()
    encoded = tokenizer(text, truncation=False, add_special_tokens=True)
    return len(encoded["input_ids"])
