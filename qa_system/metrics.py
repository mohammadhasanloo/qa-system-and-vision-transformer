"""SQuAD-style exact match and token F1.

Both metrics compare a predicted answer span against one or more gold spans after
normalisation, and take the best score over the gold set, so a question with
several acceptable answers is not penalised for picking one of them.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter

# Arabic and Persian forms that render identically but differ in code point;
# a prediction should not be marked wrong for choosing the other spelling.
CHARACTER_EQUIVALENTS = {
    "ي": "ی",  # Arabic yeh -> Persian yeh
    "ى": "ی",  # alef maksura -> Persian yeh
    "ك": "ک",  # Arabic kaf -> Persian keheh
    "‌": " ",       # zero-width non-joiner -> space
}
PERSIAN_PUNCTUATION = "،؛؟«»٫٬"


def normalise(text: str) -> str:
    """Lowercase, strip punctuation and articles, and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).lower()
    for source, target in CHARACTER_EQUIVALENTS.items():
        text = text.replace(source, target)
    text = "".join(" " if c in string.punctuation + PERSIAN_PUNCTUATION else c for c in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Combining marks are optional diacritics in Persian and Arabic script.
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.split())


def exact_match(prediction: str, references: list[str] | str) -> float:
    """1.0 if the prediction matches any reference after normalisation."""
    references = [references] if isinstance(references, str) else references
    return float(any(normalise(prediction) == normalise(r) for r in references))


def _f1(prediction: str, reference: str) -> float:
    predicted_tokens = normalise(prediction).split()
    reference_tokens = normalise(reference).split()

    # An empty gold answer ("unanswerable") is only right if nothing is predicted.
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)

    shared = Counter(predicted_tokens) & Counter(reference_tokens)
    overlap = sum(shared.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, references: list[str] | str) -> float:
    """Best token-overlap F1 against any reference."""
    references = [references] if isinstance(references, str) else references
    return max(_f1(prediction, r) for r in references)


def score(predictions: list[str], references: list[list[str] | str]) -> dict[str, float]:
    """Corpus-level exact match and F1, as percentages."""
    if len(predictions) != len(references):
        raise ValueError(
            f"{len(predictions)} predictions against {len(references)} references"
        )
    if not predictions:
        return {"exact_match": 0.0, "f1": 0.0}
    return {
        "exact_match": 100 * sum(map(exact_match, predictions, references)) / len(predictions),
        "f1": 100 * sum(map(token_f1, predictions, references)) / len(predictions),
    }
