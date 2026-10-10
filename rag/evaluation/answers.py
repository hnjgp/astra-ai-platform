"""
Simple, transparent answer metrics.

These are lexical heuristics, not semantic judgements: they are cheap,
deterministic and good for catching regressions. An LLM-as-judge or
human review is needed for real quality claims.
"""

import re


STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "this", "that", "with",
    "from", "have", "has", "had", "not", "but", "you", "your", "they",
    "their", "there", "what", "which", "when", "where", "who", "how",
    "will", "would", "should", "could", "can", "must", "per", "into",
    "about", "than", "then", "them", "these", "those", "been", "being",
    "also", "only", "each", "every", "any", "all", "its", "our",
}

CITATION = re.compile(r"\[Source \d+\]", re.IGNORECASE)


def content_words(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"\w+", text.lower())
        if len(word) > 2 and word not in STOPWORDS
    }


def keyword_correctness(
    answer: str,
    expected_keywords: list[str],
) -> float:
    """
    Share of expected keywords that appear in the answer.
    """

    if not expected_keywords:
        raise ValueError(
            "expected_keywords must not be empty"
        )

    lowered = answer.lower()

    found = sum(
        keyword.lower() in lowered
        for keyword in expected_keywords
    )

    return found / len(expected_keywords)


def groundedness(
    answer: str,
    context: str,
) -> float:
    """
    Share of the answer's content words that also appear in the
    retrieved context. Low values hint at unsupported content.
    """

    answer_words = content_words(CITATION.sub("", answer))

    if not answer_words:
        return 0.0

    return len(answer_words & content_words(context)) / len(answer_words)


def has_citation(answer: str) -> bool:
    return bool(CITATION.search(answer))
