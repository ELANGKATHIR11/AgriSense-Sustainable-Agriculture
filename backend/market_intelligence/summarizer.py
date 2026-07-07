# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

import logging
import re
from backend.llm.agri_assistant import chat_query_ollama

logger = logging.getLogger("MarketSummarizer")


def classify_gov_category(title: str, text: str) -> str:
    """
    Step 16: Categorize updates into specific bins.
    """
    combined = (title + " " + text).lower()

    if any(k in combined for k in ["msp", "minimum support price", "support price"]):
        return "MSP"
    if any(
        k in combined for k in ["subsidy", "subsidies", "fertilizer subsidy", "grant"]
    ):
        return "Subsidies"
    if any(
        k in combined
        for k in [
            "weather advisory",
            "rain",
            "monsoon",
            "meteorological",
            "imd",
            "advisory",
            "cyclone",
        ]
    ):
        return "Weather Advisories"
    if any(
        k in combined for k in ["export", "import", "trade", "tariff", "export duty"]
    ):
        return "Exports/Imports"
    if any(
        k in combined
        for k in ["loan", "credit", "kcc", "interest subvention", "finance", "debt"]
    ):
        return "Loans"
    if any(
        k in combined for k in ["insurance", "pmfby", "claim", "crop damage", "payout"]
    ):
        return "Insurance"
    if any(
        k in combined
        for k in ["policy", "cabinet approve", "bill", "act", "regulation"]
    ):
        return "Policy"

    return "Schemes"


def get_word_set(text: str) -> set[str]:
    """Clean text and return set of significant words."""
    words = re.findall(r"\w+", text.lower())
    # Remove small common words
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "of",
        "is",
        "are",
        "was",
        "were",
    }
    return {w for w in words if len(w) > 3 and w not in stopwords}


def is_similar_article(title1: str, title2: str, threshold: float = 0.4) -> bool:
    """
    Deduplication / Clustering: calculate Jaccard similarity between two titles.
    """
    words1 = get_word_set(title1)
    words2 = get_word_set(title2)
    if not words1 or not words2:
        return False
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union) >= threshold


async def summarize_update(title: str, raw_text: str, category: str) -> str:
    """
    Summarize agricultural update or news into farmer-friendly language using AgriGPT.
    """
    prompt = (
        f"You are AgriGPT, a helpful agricultural advisor. Summarize the following "
        f"{category} update into extremely simple, concise, farmer-friendly language (maximum 2-3 sentences).\n\n"
        f"Title: {title}\n"
        f"Content: {raw_text}\n\n"
        f"Provide only the direct summary without conversational preambles."
    )

    try:
        summary = await chat_query_ollama(prompt)
        summary = summary.strip().strip('"').strip("'")
        if summary.startswith("Thank you for asking") or "explanation:" in summary:
            return fallback_summarize(title, raw_text)
        return summary
    except Exception as e:
        logger.warning(f"Failed to use AgriGPT for summary, using fallback: {e}")
        return fallback_summarize(title, raw_text)


def fallback_summarize(title: str, text: str) -> str:
    """
    A simple fallback text summarizer when LLM is offline.
    """
    if not text:
        return title
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if len(sentences) >= 2:
        summary = f"{sentences[0]}. {sentences[1]}."
    else:
        summary = text[:180] + "..." if len(text) > 180 else text
    return f"{title}: {summary}"
