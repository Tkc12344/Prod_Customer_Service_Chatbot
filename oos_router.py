"""
oos_router.py — Contextual OOS Pipeline
========================================

Architecture
------------
Every out-of-scope (OOS) query passes through a four-stage contextual pipeline:

  Stage 0 — Semantic fast-path (Qdrant / TF-IDF KB search)
      Checks the internal knowledge base first.  If a high-confidence policy
      match is found the query is treated as in_scope_edge and answered
      immediately without any LLM or web call.

  Stage 1 — Gemini contextual analysis  (ALL OOS queries)
      Gemini reads the query and any available KB context, then returns a
      structured JSON classification:
        { "oos_type": "...", "reasoning": "...", "search_query": "..." }
      This is the single source of truth for routing decisions.

  Stage 2 — Routing based on Gemini's classification
      live_data        → SerpAPI web RAG → Gemini fact extraction → Mistral writes
      general_knowledge→ SerpAPI web RAG → Mistral writes  (falls back to Mistral knowledge)
      unknown_intent   → SerpAPI web RAG → Mistral writes  (falls back to static redirect)
      in_scope_edge    → KB semantic search → Mistral writes
      ambiguous        → Gemini generates a clarifying question
      chitchat         → static warm redirect (no LLM call)
      policy_restricted→ static refusal (no LLM call)

  Stage 3 — Mistral answer writer  (live_data, general_knowledge, unknown_intent, in_scope_edge)
      Receives structured facts from the web RAG retrieval (or KB) and writes
      the final customer-facing response grounded in retrieved documents.

SerpAPI as a RAG retrieval tool
--------------------------------
SerpAPI is treated as a live document retriever, not a last resort.  For any
query that is not answered by the internal KB, the pipeline:
  1. Sends a refined search query to SerpAPI (Google Search)
  2. Parses the raw results into structured snippets (answer_box + organic)
  3. Passes those snippets to Gemini for fact extraction
  4. Feeds the extracted facts to Mistral as the retrieved context
  5. Mistral synthesises a grounded answer from those facts only

This means the bot can answer general_knowledge and unknown_intent queries
accurately using live web data rather than hallucinating from training weights.

OOS type priority (Gemini decides, patterns are a fast pre-filter only):
  1. policy_restricted  — harmful / inappropriate
  2. in_scope_edge      — borderline support question
  3. chitchat           — jokes, small talk
  4. live_data          — needs real-time data: weather, news, prices, scores
  5. general_knowledge  — factual/complex question (web RAG first, then knowledge)
  6. ambiguous          — too vague to route
  7. unknown_intent     — catch-all (web RAG first, then static redirect)
"""

import os
import re
import json
import logging
import random
import concurrent.futures
from typing import Optional, Dict

logger = logging.getLogger(__name__)

SUPPORT_SCOPE = (
    "orders, returns, refunds, billing, payments, shipping, "
    "account help, technical support, or appointments"
)

# ── Fast pre-filter patterns (avoid LLM call for obvious cases) ───────────────

_RESTRICTED_PATTERNS = [
    r"\b(hack|exploit|bypass|crack|steal|fraud|scam|phish)\b",
    r"\b(medical advice|diagnos|prescri|drug dosage|self.harm|suicid)\b",
    r"\b(legal advice|sue|lawsuit|attorney|lawyer)\b",
    r"\b(financial advice|invest|stock tip|crypto|forex)\b",
    r"\b(personal relationship|dating advice|breakup|divorce)\b",
    r"\b(write.*essay|do.*homework|cheat|plagiar)\b",
]

_CHITCHAT_PATTERNS = [
    r"\b(tell me a joke|make me laugh|funny|riddle|pun)\b",
    r"\b(bored|entertain me|sing|dance|play a game|quiz me)\b",
]

_LIVE_DATA_PATTERNS = [
    r"\b(weather|forecast|temperature|rain|sunny|cloudy|humidity|wind speed)\b",
    r"\b(current time|what time is it|time in|local time)\b",
    r"\b(latest news|breaking news|current events|headlines)\b",
    r"\b(live score|match result|who won|final score|standings)\b",
    r"\b(exchange rate|currency rate|usd to|eur to|gbp to|zar to|rand to)\b",
    r"\b(stock price|share price|market cap|nasdaq|nyse|jse)\b",
    r"\b(current price of|how much does .* cost today)\b",
    r"\b(population of|gdp of|capital of|president of|prime minister of|leader of)\b",
]

_IN_SCOPE_EDGE_PATTERNS = [
    r"\b(warranty|guarantee|product quality|defective|broken item)\b",
    r"\b(loyalty|reward|points|voucher|coupon|promo|discount code)\b",
    r"\b(store|branch|location|opening hours|contact number)\b",
    r"\b(gift card|gift wrap|gift message)\b",
]

_AMBIGUOUS_PATTERNS = [
    r"^.{0,10}$",
    r"^\W+$",
    r"\?{2,}",
    r"\b(idk|not sure|maybe|kind of|sort of|dunno)\b",
]

# ── Static response pools ─────────────────────────────────────────────────────

_CHITCHAT_REDIRECTS = [
    "Ha, I wish! I'm a support assistant — best at orders, returns, billing, and shipping. Anything I can help with?",
    "That's a bit outside my lane! I'm here for support questions — orders, billing, shipping, and more.",
    "I'll leave the entertainment to others 😄 — I'm your support assistant. Got a question about an order or account?",
    "Not quite my specialty! I'm focused on customer support. Is there something I can help you with today?",
]

_FALLBACK_REDIRECTS = [
    "I'm your support assistant — I can help with orders, returns, billing, shipping, and account questions. What do you need?",
    "That's outside what I can help with, but I'm great at support questions — orders, billing, shipping. What can I do for you?",
    "I'm not sure I can help with that. I specialise in customer support — orders, returns, payments, and more. Anything I can assist with?",
]

_REFUSE_RESPONSES = [
    "I'm not able to help with that. If you have a question about an order, billing, or your account, I'm happy to assist.",
    "That's not something I can help with. I'm here for support questions — orders, returns, billing, and shipping.",
]

_CLARIFY_RESPONSES = [
    "Could you give me a bit more detail? Is this about an order, a payment, your account, or something else?",
    "I want to make sure I help you with the right thing — is this related to an order, a return, billing, or something else?",
    "Can you tell me a bit more? For example, is this about a delivery, a charge, or your account?",
]

# ── Config ────────────────────────────────────────────────────────────────────

_MAX_WORDS     = int(os.getenv("OOS_MAX_RESPONSE_WORDS", "50"))
OOS_MAX_TOKENS = int(os.getenv("OOS_MAX_TOKENS", "100"))

_PREAMBLE_RE = re.compile(
    r"^(of course[,!]?\s*|sure[,!]?\s*|certainly[,!]?\s*|absolutely[,!]?\s*"
    r"|great question[,!]?\s*|happy to help[,!]?\s*|i'd be happy to[,!]?\s*"
    r"|no problem[,!]?\s*|thanks for (reaching out|contacting us)[,!]?\s*)",
    re.IGNORECASE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Strip preamble, capitalise, truncate to word limit."""
    if not text:
        return text
    text = _PREAMBLE_RE.sub("", text).strip()
    if text:
        text = text[0].upper() + text[1:]
    words = text.split()
    if len(words) > _MAX_WORDS:
        truncated = " ".join(words[:_MAX_WORDS])
        last_end  = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
        text      = truncated[:last_end + 1] if last_end > 20 else truncated + "..."
    return text.strip()


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$",       "", text, flags=re.MULTILINE)
    return text.strip()


def _rand(pool: list) -> str:
    return random.choice(pool)


def _pattern_match(patterns: list, text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


# ── LLM accessors ─────────────────────────────────────────────────────────────

_lc_gemini = None


def _get_gemini():
    """Lazy-init LangChain Gemini wrapper (contextual analyser).
    Tries GEMINI_MODEL first; falls back to GEMINI_FALLBACK_MODEL on init failure.
    """
    global _lc_gemini
    if _lc_gemini is None:
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            return None

        models_to_try = [
            os.getenv("GEMINI_MODEL",          "gemini-1.5-flash"),
            os.getenv("GEMINI_FALLBACK_MODEL",  "gemini-2.0-flash"),
        ]
        # Deduplicate while preserving order
        seen = set()
        models_to_try = [m for m in models_to_try if m and not (m in seen or seen.add(m))]

        for model_name in models_to_try:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                _lc_gemini = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=key,
                    temperature=0.1,
                    convert_system_message_to_human=True,
                    max_retries=2,
                )
                logger.info(f"[Gemini] contextual analyser ready — {model_name}")
                break
            except Exception as e:
                logger.warning(f"[Gemini] init failed for {model_name}: {e}")

    return _lc_gemini


def _gemini_invoke(messages: list) -> Optional[str]:
    """
    Invoke Gemini with automatic model fallback on overload/429/503.
    Returns response content string or None.
    """
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        return None

    models_to_try = [
        os.getenv("GEMINI_MODEL",         "gemini-1.5-flash"),
        os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.0-flash"),
    ]
    seen = set()
    models_to_try = [m for m in models_to_try if m and not (m in seen or seen.add(m))]

    _overload_kw = ("429", "503", "quota", "rate", "overload", "capacity",
                    "resource_exhausted", "too many", "high volume", "high traffic")

    for model_name in models_to_try:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=key,
                temperature=0.1,
                convert_system_message_to_human=True,
                max_retries=1,
            )
            resp = llm.invoke(messages)
            text = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
            if text:
                return text
        except Exception as e:
            err = str(e).lower()
            if any(kw in err for kw in _overload_kw):
                logger.warning(f"[Gemini] {model_name} overloaded — trying next model: {e}")
                continue
            logger.warning(f"[Gemini] {model_name} failed: {e}")
            return None

    logger.warning("[Gemini] all models failed or overloaded")
    return None


# ── Stage 0: Semantic fast-path (TF-IDF cosine KB search) ────────────────────

def _semantic_kb_search(query: str, rag_engine) -> Optional[str]:
    """
    Stage 0 fast-path: pure dense vector search via Qdrant.
    Falls back to hybrid TF-IDF search if Qdrant is unavailable.
    Returns the top policy text if relevance >= threshold, else None.
    """
    if not rag_engine:
        return None
    try:
        # Use semantic_search (Qdrant dense) when available, else hybrid fallback
        if hasattr(rag_engine.knowledge_base, "semantic_search"):
            results = rag_engine.knowledge_base.semantic_search(query, top_k=1)
        else:
            results = rag_engine.knowledge_base.retrieve_relevant_policies(query, top_k=1)

        if results and results[0]["relevance"] >= 0.35:
            logger.info(
                f"[Semantic] KB hit: {results[0]['name']} "
                f"relevance={results[0]['relevance']:.2f}"
            )
            return results[0]["policy"]
    except Exception as e:
        logger.warning(f"[Semantic] KB search failed: {e}")
    return None


# ── Stage 1: Gemini contextual analysis ──────────────────────────────────────

def _gemini_analyse(query: str, kb_ctx: Optional[str] = None) -> Dict:
    """
    Ask Gemini to classify the OOS query and extract key facts.

    Returns a dict:
      {
        "oos_type":     one of the 7 labels,
        "reasoning":    short explanation,
        "key_facts":    extracted facts (for live_data / in_scope_edge),
        "answer_hint":  one-sentence answer hint,
        "search_query": refined query for SerpAPI (live_data only),
      }
    Falls back to pattern-based classification if Gemini is unavailable.
    """
    llm = _get_gemini()

    context_block = ""
    if kb_ctx:
        context_block = f"\n\nKNOWLEDGE BASE CONTEXT:\n{kb_ctx}"

    if llm:
        prompt = (
            "You are a contextual analyser for a customer support chatbot.\n"
            "The bot handles ONLY: " + SUPPORT_SCOPE + ".\n\n"
            "Analyse the customer message below and return a JSON object with these fields:\n"
            "  oos_type     — ONE of: policy_restricted | in_scope_edge | chitchat | "
            "live_data | general_knowledge | ambiguous | unknown_intent\n"
            "  reasoning    — one sentence explaining your classification\n"
            "  key_facts    — any relevant facts extracted from context (empty string if none)\n"
            "  answer_hint  — one clear sentence that would answer the question (empty if N/A)\n"
            "  search_query — a refined web search query (only for live_data, else empty)\n\n"
            "Classification guide:\n"
            "  policy_restricted  — harmful, illegal, or inappropriate request\n"
            "  in_scope_edge      — borderline support question (warranty, loyalty, store info)\n"
            "  chitchat           — jokes, small talk, entertainment\n"
            "  live_data          — needs real-time data: weather, news, prices, scores\n"
            "  general_knowledge  — factual/complex question answerable from training knowledge\n"
            "  ambiguous          — too vague to classify confidently\n"
            "  unknown_intent     — none of the above\n"
            + context_block + "\n\n"
            "Customer message: \"" + query + "\"\n\n"
            "Return ONLY valid JSON, no markdown fences."
        )
        try:
            from langchain_core.messages import HumanMessage
            raw = _gemini_invoke([HumanMessage(content=prompt)])
            if raw:
                data = json.loads(_strip_fences(raw))
                # Normalise
                valid_types = {
                    "policy_restricted", "in_scope_edge", "chitchat",
                    "live_data", "general_knowledge", "ambiguous", "unknown_intent",
                }
                if data.get("oos_type") not in valid_types:
                    data["oos_type"] = "unknown_intent"
                data.setdefault("reasoning",    "")
                data.setdefault("key_facts",    "")
                data.setdefault("answer_hint",  "")
                data.setdefault("search_query", query)
                logger.info(
                    f"[Gemini] analysis: type={data['oos_type']} "
                    f"hint={data['answer_hint'][:50]}"
                )
                return data
        except Exception as e:
            logger.warning(f"[Gemini] analysis failed: {e} — falling back to patterns")

    # ── Pattern fallback (no Gemini) ─────────────────────────────────────────
    return _pattern_classify(query)


def _pattern_classify(query: str) -> Dict:
    """Fast pattern-based fallback when Gemini is unavailable."""
    q = query.lower().strip()
    base = {"reasoning": "pattern", "key_facts": "", "answer_hint": "", "search_query": query}

    if _pattern_match(_RESTRICTED_PATTERNS,  q): return {**base, "oos_type": "policy_restricted"}
    if _pattern_match(_IN_SCOPE_EDGE_PATTERNS, q): return {**base, "oos_type": "in_scope_edge"}
    if _pattern_match(_CHITCHAT_PATTERNS,     q): return {**base, "oos_type": "chitchat"}
    if _pattern_match(_LIVE_DATA_PATTERNS,    q): return {**base, "oos_type": "live_data"}
    if _pattern_match(_AMBIGUOUS_PATTERNS,    q): return {**base, "oos_type": "ambiguous"}
    return {**base, "oos_type": "unknown_intent"}


# ── Stage 2a: SerpAPI web retrieval (RAG document source) ────────────────────

def _serp_search(query: str) -> Optional[str]:
    """
    Retrieve web documents via SerpAPI (primary) or DuckDuckGo (fallback).

    Treats SerpAPI as a RAG document retriever:
      - answer_box  → highest-priority direct answer
      - knowledge_graph → structured entity facts
      - organic results → supporting snippets with titles and URLs

    Returns a structured document string ready to be passed to Gemini for
    fact extraction, or None if all providers fail.
    """
    serp_key = (
        os.getenv("SERPAPI_API_KEY", "").strip() or
        os.getenv("SERPAPI_KEY",     "").strip()
    )

    if serp_key:
        # ── Primary: direct SerpAPI SDK (structured retrieval) ────────────────
        try:
            from serpapi import GoogleSearch
            results = GoogleSearch({
                "q":       query,
                "api_key": serp_key,
                "num":     5,
                "hl":      "en",
                "gl":      "us",
            }).get_dict()

            docs = []

            # Tier 1 — answer box (direct answer, highest confidence)
            if "answer_box" in results:
                ab  = results["answer_box"]
                ans = ab.get("answer") or ab.get("snippet") or ab.get("result")
                if ans:
                    docs.append(f"[DIRECT ANSWER]\n{ans.strip()}")

            # Tier 2 — knowledge graph (structured entity facts)
            if "knowledge_graph" in results:
                kg = results["knowledge_graph"]
                kg_parts = []
                if kg.get("title"):
                    kg_parts.append(kg["title"])
                if kg.get("description"):
                    kg_parts.append(kg["description"])
                for k, v in kg.items():
                    if k not in ("title", "description", "header_images",
                                 "images", "source", "type") and isinstance(v, str):
                        kg_parts.append(f"{k}: {v}")
                if kg_parts:
                    docs.append("[KNOWLEDGE GRAPH]\n" + "\n".join(kg_parts[:6]))

            # Tier 3 — organic results (supporting snippets)
            for r in results.get("organic_results", [])[:5]:
                snippet = (r.get("snippet") or "").strip()
                title   = (r.get("title")   or "").strip()
                link    = (r.get("link")    or "").strip()
                if snippet:
                    source = f" ({title})" if title else ""
                    docs.append(f"[WEB RESULT{source}]\n{snippet}")

            if docs:
                combined = "\n\n".join(docs)
                logger.info(
                    f"[SerpAPI] retrieved {len(docs)} document(s) "
                    f"({len(combined)} chars) for: '{query[:50]}'"
                )
                return combined[:3000]

        except Exception as e:
            logger.warning(f"[SerpAPI] SDK retrieval failed: {e}")

        # ── Secondary: LangChain SerpAPI wrapper ──────────────────────────────
        try:
            from langchain_community.utilities import SerpAPIWrapper
            os.environ["SERPAPI_API_KEY"] = serp_key
            raw = SerpAPIWrapper(
                serpapi_api_key=serp_key,
                params={"num": 5, "hl": "en", "gl": "us"},
            ).run(query)
            if raw and len(raw.strip()) > 20:
                logger.info(f"[SerpAPI] LangChain fallback: {len(raw)} chars")
                return raw[:3000]
        except Exception as e:
            logger.warning(f"[SerpAPI] LangChain fallback failed: {e}")

    # ── Tertiary: DuckDuckGo (no API key required) ────────────────────────────
    try:
        from langchain_community.tools import DuckDuckGoSearchResults
        raw  = DuckDuckGoSearchResults(num_results=5).run(query)
        hits = re.findall(r"\[snippet:\s*(.*?),\s*title:\s*(.*?),\s*link:\s*(.*?)\]",
                          raw, re.DOTALL)
        if hits:
            docs = [
                f"[WEB RESULT ({title.strip()})]\n{snippet.strip()}"
                for snippet, title, _ in hits[:5]
                if snippet.strip()
            ]
            if docs:
                combined = "\n\n".join(docs)
                logger.info(f"[DuckDuckGo] retrieved {len(docs)} result(s)")
                return combined[:3000]
        if raw and len(raw.strip()) > 20:
            return raw[:3000]
    except Exception as e:
        logger.warning(f"[DuckDuckGo] failed: {e}")

    logger.warning("[WebRetrieval] all providers failed — no documents retrieved")
    return None


# ── Stage 2b: Gemini fact extraction from retrieved web documents ─────────────

def _gemini_extract_facts(query: str, retrieved_docs: str) -> Dict:
    """
    Ask Gemini to extract structured facts from retrieved web documents.

    Treats the retrieved_docs string as a RAG context window — Gemini reads
    the documents and extracts only the facts needed to answer the query.

    Returns {"key_facts": "...", "answer_hint": "...", "has_answer": bool}.
    """
    llm = _get_gemini()
    if not llm:
        # No Gemini — pass raw docs directly to Mistral as context
        return {
            "key_facts":  retrieved_docs[:1200],
            "answer_hint": "",
            "has_answer":  True,
        }

    prompt = (
        "You are a fact extractor for a RAG pipeline.\n"
        "Given the RETRIEVED DOCUMENTS below, extract only the facts needed "
        "to answer the QUESTION accurately.\n\n"
        "QUESTION: " + query + "\n\n"
        "RETRIEVED DOCUMENTS:\n" + retrieved_docs + "\n\n"
        "Return JSON only — no markdown fences:\n"
        "{\n"
        "  \"key_facts\":  \"the most relevant facts, numbers, or data from the documents\",\n"
        "  \"answer_hint\": \"one clear sentence directly answering the question\",\n"
        "  \"has_answer\":  true or false — whether the documents contain a usable answer\n"
        "}\n\n"
        "If the documents do not contain a relevant answer, set has_answer to false "
        "and leave key_facts and answer_hint empty."
    )
    try:
        from langchain_core.messages import HumanMessage
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw  = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
        data = json.loads(_strip_fences(raw))
        data.setdefault("key_facts",   "")
        data.setdefault("answer_hint", "")
        data.setdefault("has_answer",  bool(data.get("key_facts")))
        logger.info(
            f"[Gemini] facts extracted — has_answer={data['has_answer']} "
            f"facts='{data['key_facts'][:60]}'"
        )
        return data
    except Exception as e:
        logger.warning(f"[Gemini] fact extraction failed: {e}")
        return {"key_facts": retrieved_docs[:800], "answer_hint": "", "has_answer": True}


# ── Stage 3: Mistral answer writer ───────────────────────────────────────────

def _mistral_write(
    query:          str,
    key_facts:      str  = "",
    answer_hint:    str  = "",
    soft_redirect:  bool = True,
    from_knowledge: bool = False,
    web_grounded:   bool = False,
) -> Optional[str]:
    """
    Mistral writes the final customer-facing answer.

    Modes
    -----
      web_grounded=True    — facts came from SerpAPI web retrieval; Mistral must
                             answer ONLY from those facts (no hallucination)
      from_knowledge=True  — no external context; Mistral uses its training data
      key_facts provided   — Mistral synthesises from structured facts (search / KB)
    """
    ms = _get_mistral()
    if not ms.enabled:
        return None

    redirect = (
        " After answering, add one short friendly sentence offering support help "
        "(e.g. 'Let me know if you need help with an order, billing, or shipping.')."
        if soft_redirect else ""
    )

    if web_grounded and key_facts:
        # Strict grounding — answer must come from retrieved web documents only
        system = (
            "You are a helpful assistant. "
            "Answer using ONLY the RETRIEVED FACTS provided below. "
            "Do NOT add information from your training data. "
            "Be direct and accurate — 1–2 sentences max. No preamble."
            + redirect
        )§
        hint_line = f"SUGGESTED ANSWER: {answer_hint}\n\n" if answer_hint else ""
        user_msg  = (
            f"RETRIEVED FACTS (from web search):\n{key_facts}\n\n"
            f"{hint_line}"
            f"Question: {query}\n\nAnswer:"
        )

    elif from_knowledge:
        system   = (
            "You are a knowledgeable assistant. "
            "Answer the question in 1–2 sentences using your own knowledge. "
            "Be accurate and concise. No preamble." + redirect
        )
        user_msg = f"Question: {query}\n\nAnswer:"

    elif key_facts:
        system   = (
            "You are a helpful assistant. "
            "Answer using ONLY the KEY FACTS provided. "
            "Be direct — 1–2 sentences max. No preamble." + redirect
        )
        hint_line = f"SUGGESTED ANSWER: {answer_hint}\n\n" if answer_hint else ""
        user_msg  = f"KEY FACTS: {key_facts}\n\n{hint_line}Question: {query}\n\nAnswer:"

    else:
        system   = (
            "You are a knowledgeable assistant. "
            "Answer in 1–2 sentences. No preamble." + redirect
        )
        user_msg = f"Question: {query}\n\nAnswer:"

    try:
        result = ms.chat(
            [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.4,
            max_tokens=OOS_MAX_TOKENS,
        )
        if result:
            logger.info(f"[Mistral] answer: {len(result)} chars")
            return result
    except Exception as e:
        logger.warning(f"[Mistral] write failed: {e}")
    return None


# ── Pipeline runners ──────────────────────────────────────────────────────────

def _run_web_rag_pipeline(
    query:        str,
    search_query: str,
    soft_redirect: bool = True,
    fallback_fn   = None,
) -> str:
    """
    Shared SerpAPI-as-RAG pipeline used by live_data, general_knowledge,
    and unknown_intent.

    Flow
    ----
      1. Retrieve web documents via SerpAPI (or DuckDuckGo fallback)
      2. Gemini extracts structured facts from the retrieved documents
      3. Mistral writes a grounded answer from those facts only
      4. If retrieval fails or Gemini says has_answer=False, call fallback_fn()

    Parameters
    ----------
    query         : original customer question
    search_query  : refined query for the search engine (from Gemini analysis)
    soft_redirect : whether Mistral should append a support offer
    fallback_fn   : callable() → str  used when web retrieval yields nothing
    """
    logger.info(f"[WebRAG] retrieving documents for: '{query[:60]}'")

    # Step 1 — retrieve
    retrieved_docs = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_serp_search, search_query or query)
        try:
            retrieved_docs = future.result(timeout=8)
        except Exception as e:
            logger.warning(f"[WebRAG] retrieval timeout/error: {e}")

    if not retrieved_docs:
        logger.info("[WebRAG] no documents retrieved — using fallback")
        return fallback_fn() if fallback_fn else _rand(_FALLBACK_REDIRECTS)

    # Step 2 — extract facts from retrieved documents
    facts = _gemini_extract_facts(query, retrieved_docs)

    if not facts.get("has_answer") or not facts.get("key_facts"):
        logger.info("[WebRAG] retrieved docs contain no usable answer — using fallback")
        return fallback_fn() if fallback_fn else _rand(_FALLBACK_REDIRECTS)

    # Step 3 — Mistral writes grounded answer
    answer = _mistral_write(
        query,
        key_facts=facts["key_facts"],
        answer_hint=facts.get("answer_hint", ""),
        soft_redirect=soft_redirect,
        web_grounded=True,
    )
    if answer:
        logger.info("[WebRAG] answer written from retrieved web documents")
        return _clean(answer)

    # Step 4 — Mistral failed; use Gemini's answer_hint directly
    if facts.get("answer_hint"):
        suffix = " Let me know if you need help with an order or account." if soft_redirect else ""
        return _clean(facts["answer_hint"] + suffix)

    return fallback_fn() if fallback_fn else _rand(_FALLBACK_REDIRECTS)


def _run_live_data_pipeline(query: str, search_query: str) -> str:
    """
    live_data path — real-time data (weather, prices, scores, news).
    Uses the shared web RAG pipeline; falls back to Mistral training knowledge.
    """
    logger.info(f"[Pipeline:live_data] '{query[:50]}'")
    return _run_web_rag_pipeline(
        query=query,
        search_query=search_query or query,
        soft_redirect=True,
        fallback_fn=lambda: (
            _clean(_mistral_write(query, from_knowledge=True, soft_redirect=True))
            or _rand(_FALLBACK_REDIRECTS)
        ),
    )


def _run_general_knowledge_pipeline(query: str, analysis: Dict) -> str:
    """
    general_knowledge path — factual questions the bot was not trained on.

    Priority:
      1. SerpAPI web RAG (retrieved documents → grounded answer)
      2. Gemini answer_hint (if already extracted during analysis)
      3. Mistral training knowledge (last resort)
    """
    logger.info(f"[Pipeline:general_knowledge] '{query[:50]}'")

    # Use Gemini's refined search_query if available, else the raw query
    search_query = analysis.get("search_query") or query

    def _knowledge_fallback() -> str:
        hint = analysis.get("answer_hint", "")
        if hint:
            answer = _mistral_write(
                query, key_facts=hint, answer_hint="", soft_redirect=True,
            )
            if answer:
                return _clean(answer)
        answer = _mistral_write(query, from_knowledge=True, soft_redirect=True)
        return _clean(answer) if answer else _rand(_FALLBACK_REDIRECTS)

    return _run_web_rag_pipeline(
        query=query,
        search_query=search_query,
        soft_redirect=True,
        fallback_fn=_knowledge_fallback,
    )


def _run_unknown_intent_pipeline(query: str, analysis: Dict) -> str:
    """
    unknown_intent path — query doesn't match any known category.

    Priority:
      1. SerpAPI web RAG (attempt to answer from retrieved documents)
      2. Static redirect (if web retrieval yields nothing useful)
    """
    logger.info(f"[Pipeline:unknown_intent] '{query[:50]}'")

    search_query = analysis.get("search_query") or query

    return _run_web_rag_pipeline(
        query=query,
        search_query=search_query,
        soft_redirect=True,
        fallback_fn=lambda: _rand(_FALLBACK_REDIRECTS),
    )


def _run_in_scope_edge_pipeline(query: str, kb_ctx: str, analysis: Dict) -> str:
    """
    in_scope_edge path — borderline support question answered from KB.
    Semantic KB search already done → Gemini's key_facts + Mistral writes.
    """
    logger.info(f"[Pipeline:in_scope_edge] '{query[:50]}'")
    key_facts   = analysis.get("key_facts",   "") or kb_ctx[:600]
    answer_hint = analysis.get("answer_hint", "")
    answer = _mistral_write(
        query,
        key_facts=key_facts,
        answer_hint=answer_hint,
        soft_redirect=False,
    )
    if answer:
        return _clean(answer)
    if answer_hint:
        return _clean(answer_hint)
    return kb_ctx[:300]


def _run_ambiguous_pipeline(query: str) -> str:
    """
    ambiguous path — Gemini generates a short clarifying question.
    """
    llm = _get_gemini()
    if llm:
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            resp = llm.invoke([
                SystemMessage(content=(
                    "You are a customer support assistant. "
                    "Ask ONE short clarifying question (max 20 words) to find out "
                    "if the user needs help with " + SUPPORT_SCOPE + ". No preamble."
                )),
                HumanMessage(content=query),
            ])
            text = resp.content.strip() if hasattr(resp, "content") else ""
            if text:
                return _clean(text)
        except Exception as e:
            logger.warning(f"[Ambiguous] Gemini clarify failed: {e}")
    return _rand(_CLARIFY_RESPONSES)


# ── OOS Router ────────────────────────────────────────────────────────────────

class OOSRouter:
    """
    Contextual OOS router.

    Every query goes through:
      Stage 0 — Semantic KB fast-path (TF-IDF cosine)
      Stage 1 — Gemini contextual analysis
      Stage 2 — Route to the appropriate pipeline
      Stage 3 — Mistral writes the final answer (where applicable)
    """

    def __init__(self, rag_engine=None):
        self.rag_engine = rag_engine
        _get_gemini()   # warm up connection at startup
        logger.info(
            "OOSRouter ready — pipeline: "
            "KB semantic → Gemini analysis → SerpAPI web RAG → Mistral write"
        )

    def route(self, query: str, intent: str = "out_of_scope", confidence: float = 0.0) -> tuple:
        """
        Route an OOS query through the contextual pipeline.
        Returns (response_text, oos_type).

        Routing table
        -------------
          policy_restricted  → static refusal
          chitchat           → static redirect
          in_scope_edge      → KB context → Mistral (grounded on KB)
          ambiguous          → Gemini clarifying question
          live_data          → SerpAPI web RAG → Mistral (grounded on web docs)
          general_knowledge  → SerpAPI web RAG → Mistral (grounded on web docs)
                               fallback: Gemini hint → Mistral training knowledge
          unknown_intent     → SerpAPI web RAG → Mistral (grounded on web docs)
                               fallback: static redirect
        """
        logger.info(f"[OOSRouter] '{query[:60]}' intent={intent} conf={confidence:.2f}")

        # ── Stage 0: Semantic fast-path ───────────────────────────────────────
        kb_ctx = _semantic_kb_search(query, self.rag_engine)

        # ── Stage 1: Gemini contextual analysis ───────────────────────────────
        analysis = _gemini_analyse(query, kb_ctx=kb_ctx)
        oos_type = analysis["oos_type"]
        logger.info(f"[OOSRouter] classified: {oos_type} — {analysis['reasoning'][:60]}")

        # ── Stage 2: Route ────────────────────────────────────────────────────

        if oos_type == "policy_restricted":
            return _rand(_REFUSE_RESPONSES), oos_type

        if oos_type == "chitchat":
            return _rand(_CHITCHAT_REDIRECTS), oos_type

        if oos_type == "ambiguous":
            return _run_ambiguous_pipeline(query), oos_type

        if oos_type == "in_scope_edge":
            # Use KB context already retrieved in Stage 0; if none, try a broader search
            if not kb_ctx and self.rag_engine:
                try:
                    results = self.rag_engine.knowledge_base.retrieve_relevant_policies(query, top_k=2)
                    if results:
                        kb_ctx = "\n\n".join(p["policy"] for p in results)
                except Exception as e:
                    logger.warning(f"[in_scope_edge] KB fallback failed: {e}")
            ctx = kb_ctx or analysis.get("key_facts", "")
            return _run_in_scope_edge_pipeline(query, ctx, analysis), oos_type

        if oos_type == "live_data":
            search_query = analysis.get("search_query") or query
            return _run_live_data_pipeline(query, search_query), oos_type

        if oos_type == "general_knowledge":
            # Web RAG first — fall back to Mistral training knowledge
            return _run_general_knowledge_pipeline(query, analysis), oos_type

        if oos_type == "unknown_intent":
            # Web RAG first — fall back to static redirect
            return _run_unknown_intent_pipeline(query, analysis), oos_type

        # Catch-all
        return _rand(_FALLBACK_REDIRECTS), oos_type


# ── Singleton ─────────────────────────────────────────────────────────────────

_router_instance: Optional[OOSRouter] = None


def get_router(rag_engine=None) -> OOSRouter:
    global _router_instance
    if _router_instance is None:
        _router_instance = OOSRouter(rag_engine=rag_engine)
    elif rag_engine and _router_instance.rag_engine is None:
        _router_instance.rag_engine = rag_engine
    return _router_instance


def _get_mistral():
    from llm_clients import get_mistral
    return get_mistral()
