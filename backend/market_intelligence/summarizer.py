import logging
from backend.llm.agri_assistant import chat_query_ollama

logger = logging.getLogger("MarketSummarizer")

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
        # Clean any response preambles or quotes
        summary = summary.strip().strip('"').strip("'")
        if summary.startswith("Thank you for asking") or "explanation:" in summary:
            # Fallback did not process it correctly or model offline, let's make a cleaner summary
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
    # Take first two sentences or first 180 characters
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if len(sentences) >= 2:
        summary = f"{sentences[0]}. {sentences[1]}."
    else:
        summary = text[:180] + "..." if len(text) > 180 else text
    return f"{title}: {summary}"
