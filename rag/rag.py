# rag/rag.py
"""
Retrieval-Augmented Generation (RAG) pipeline.

Explanations:
- RAG combines retrieval (semantic search) with generation (LLM).
- We retrieve relevant text chunks from FAISS and inject them into a prompt.
- Grounding: The LLM is instructed to answer ONLY using the provided context.
- Hallucination: When LLMs invent facts not present in context. In automotive systems,
  hallucinations can be dangerous. We prevent hallucination by:
  * Only providing retrieved context to the model.
  * Explicitly instructing the model to reply "Data not available" if the answer
    cannot be found in the context.
  * Optionally, verifying that the model's output contains citations or direct quotes
    from the context (not implemented here, but recommended for production).
"""

import os
from typing import List
import openai
import textwrap

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY


def build_prompt(query: str, contexts: List[str]) -> str:
    """
    Prompt template:
    - Provide context chunks.
    - Instruct the model to answer only from context.
    - If the answer is not present, respond with "Data not available".
    """
    header = (
        "You are an automotive assistant for Ford vehicles. Use ONLY the provided context "
        "to answer the user's question. Do NOT invent or add information not present in the context. "
        "If the answer cannot be found in the context, respond exactly: Data not available\n\n"
    )
    context_text = "\n\n".join([f"Context {i+1}:\n{c}" for i, c in enumerate(contexts)])
    prompt = f"{header}Context:\n{context_text}\n\nUser question: {query}\n\nAnswer concisely and only using the context."
    # Keep prompt length reasonable
    return textwrap.shorten(prompt, width=4000, placeholder=" ... [truncated]")


def generate_answer_with_openai(prompt: str) -> str:
    """
    Call OpenAI ChatCompletion. The system message reinforces grounding.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OpenAI API key not configured")

    messages = [
        {"role": "system", "content": "You are a helpful assistant that only uses provided context."},
        {"role": "user", "content": prompt}
    ]
    resp = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=300
    )
    return resp.choices[0].message.content.strip()


def generate_answer_fallback(contexts: List[str], query: str) -> str:
    """
    Simple deterministic fallback generator that strictly uses contexts.
    It searches for keywords from the query in contexts and returns matching sentences.
    If nothing matches, returns 'Data not available'.
    This ensures the strict rule: only answer using retrieved context.
    """
    combined = "\n".join(contexts)
    q_lower = query.lower()
    # Very simple heuristic: find sentences in contexts that contain any keyword from query
    keywords = [w for w in q_lower.split() if len(w) > 3]
    matched_sentences = []
    for sentence in combined.split("\n"):
        s = sentence.strip()
        if not s:
            continue
        s_lower = s.lower()
        if any(k in s_lower for k in keywords):
            matched_sentences.append(s)
    if matched_sentences:
        # Return up to 3 matched sentences joined
        return " ".join(matched_sentences[:3])
    return "Data not available"


def generate_grounded_answer(contexts: List[str], query: str) -> str:
    """
    Main RAG answer function.
    - If no contexts provided -> return "Data not available"
    - Otherwise, build prompt and call LLM if available, else fallback.
    """
    if not contexts or len(contexts) == 0:
        return "Data not available"

    prompt = build_prompt(query, contexts)
    if OPENAI_API_KEY:
        try:
            return generate_answer_with_openai(prompt)
        except Exception as e:
            # In case of API failure, use fallback
            return generate_answer_fallback(contexts, query)
    else:
        return generate_answer_fallback(contexts, query)
