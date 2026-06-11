"""
rag_system.py — Contextual RAG with Hybrid Retrieval
======================================================

Retrieval architecture
----------------------
Every query runs through a two-layer hybrid search:

  Layer 1 — TF-IDF keyword match  (sparse, fast, exact terms)
  Layer 2 — Qdrant dense vectors  (semantic, catches paraphrases & synonyms)

Final score = 0.5 × keyword_score + 0.5 × semantic_score

The dense layer uses sentence-transformers/all-MiniLM-L6-v2:
  - 22 MB model, runs fully offline, no API key required
  - Qdrant persists vectors to ./qdrant_store/ (survives restarts)
  - Falls back gracefully to TF-IDF-only if sentence-transformers
    or qdrant-client are not installed

Hybrid weights are tunable via env vars:
  HYBRID_KEYWORD_WEIGHT   default 0.5
  HYBRID_SEMANTIC_WEIGHT  default 0.5
"""

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from nltk.stem import PorterStemmer
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Hybrid weights ────────────────────────────────────────────────────────────
_KW_WEIGHT  = float(os.getenv("HYBRID_KEYWORD_WEIGHT",  "0.5"))
_SEM_WEIGHT = float(os.getenv("HYBRID_SEMANTIC_WEIGHT", "0.5"))

# Qdrant collection name and vector dimension for all-MiniLM-L6-v2
_COLLECTION   = "kb_policies"
_VECTOR_DIM   = 384
_QDRANT_PATH  = os.getenv("QDRANT_PATH", "./qdrant_store")
_QDRANT_URL   = os.getenv("QDRANT_URL", "").strip()
_QDRANT_KEY   = os.getenv("QDRANT_API_KEY", "").strip()


# ── Knowledge-base policies ───────────────────────────────────────────────────
POLICIES = {
    "order_status": {
        "keywords": [
            "order", "orders", "status", "track", "tracking", "where", "delivery", "shipped",
            "arrived", "check", "my order", "tracking number", "shipment", "package",
            "where is my order", "check order status", "track my order", "order tracking",
            "where is my package", "package status", "shipment status", "delivery status",
            "when will it arrive", "estimated delivery", "expected delivery date",
            "order not arrived", "order delayed", "late delivery", "missing package",
            "order confirmation", "order number lookup", "find my order",
            "has my order shipped", "order dispatched", "out for delivery",
            "in transit", "order processing", "track shipment", "shipping update",
            "package not received", "order still processing", "order history",
        ],
        "policy": (
            "To check your order: visit 'My Orders', enter your order number or email, "
            "and view live tracking. Standard delivery is 5–7 business days; expedited is 2–3. "
            "If tracking isn't available yet, please wait 24 hours after placing your order."
        ),
        "category": "orders",
    },
    "return_refund": {
        "keywords": [
            "return", "refund", "exchange", "money back", "send back", "replace",
            "reimbursement", "return shipping", "damaged", "wrong item", "defective",
            "how to return", "return my item", "return policy", "start a return",
            "get a refund", "request refund", "refund status", "refund timeline",
            "how long for refund", "exchange item", "wrong size", "wrong color",
            "item not as described", "damaged item", "broken item", "defective product",
            "return label", "free return", "return window", "30 day return",
            "money back guarantee", "full refund", "partial refund", "store credit",
        ],
        "policy": (
            "Returns are accepted within 30 days — items must be unused and in original packaging. "
            "Refunds process within 5–7 business days after we receive the item. "
            "To start, go to 'My Orders', select the item, and choose 'Return/Exchange'."
        ),
        "category": "orders",
    },
    "cancellation": {
        "keywords": [
            "cancel", "cancellation", "stop order", "change order", "remove", "delete",
            "cancel my order", "cancel order", "i want to cancel", "how to cancel",
            "cancel before shipping", "cancel purchase", "stop my order", "do not ship",
            "undo my order", "reverse order", "cancel immediately", "urgent cancellation",
            "can i still cancel", "is it too late to cancel", "cancel and refund",
            "cancellation fee", "cancel subscription", "cancel recurring order",
        ],
        "policy": (
            "Orders can be cancelled within 2 hours of placement before processing begins — "
            "go to 'My Orders' and click 'Cancel Order'. "
            "If it has already shipped, you'll need to initiate a return instead; "
            "refunds process in 3–5 business days."
        ),
        "category": "orders",
    },
    "billing_issue": {
        "keywords": [
            "billing", "charge", "payment", "invoice", "card", "money", "cost", "price",
            "declined", "failed", "duplicate", "unexpected", "error",
            "billing issue", "wrong charge", "incorrect charge", "overcharged",
            "duplicate charge", "charged twice", "double charge", "extra charge",
            "unauthorized charge", "payment failed", "payment declined", "card declined",
            "payment method issue", "invoice wrong", "subscription charge", "recurring charge",
            "transaction failed", "charge dispute", "billing statement",
        ],
        "policy": (
            "For duplicate or incorrect charges we'll investigate and issue a refund within 1 business day. "
            "For payment failures, please try a different card or update your payment info and retry. "
            "Share your order number or email and we'll resolve it within 24 hours."
        ),
        "category": "billing",
    },
    "technical_support": {
        "keywords": [
            "technical", "issue", "problem", "error", "bug", "crash", "not working",
            "broken", "slow", "freeze", "compatibility", "login",
            "app crashed", "app not working", "app not loading", "website down",
            "website not working", "error message", "error code", "404 error", "500 error",
            "cannot login", "cant sign in", "login not working", "account locked",
            "connection problem", "slow loading", "browser issue", "bug report",
            "system error", "not responding", "checkout not working", "search not working",
        ],
        "policy": (
            "Please try clearing your browser cache, restarting the app, or using a different browser. "
            "For login issues, reset your password and try incognito mode. "
            "If the problem persists, send us a screenshot with your device and browser details."
        ),
        "category": "support",
    },
    "account_help": {
        "keywords": [
            "account", "password", "reset", "login", "username", "profile", "settings",
            "security", "email", "two factor",
            "account help", "account issue", "account recovery", "account locked",
            "forgot password", "reset password", "password reset", "change password",
            "login problem", "two factor authentication", "2fa", "update email",
            "profile update", "delete account", "account verification", "verify identity",
            "account suspended", "account banned", "reactivate account",
        ],
        "policy": (
            "To reset your password click 'Forgot Password' on the login page and check your email "
            "(including spam). Accounts lock after 5 failed attempts — wait 15 minutes then retry. "
            "For profile or 2FA changes go to Settings → Security."
        ),
        "category": "account",
    },
    "shipping_info": {
        "keywords": [
            "shipping", "delivery", "ship", "deliver", "courier", "carrier", "express",
            "standard shipping", "free shipping", "shipping cost", "shipping fee",
            "shipping options", "overnight shipping", "same day delivery", "next day delivery",
            "international shipping", "ship to", "shipping address", "change address",
            "shipping time", "when will it ship", "dispatch date", "estimated arrival",
        ],
        "policy": (
            "Standard shipping (5–7 business days) is free on orders over $35, otherwise $4.99. "
            "Expedited (2–3 days) costs $9.99, and overnight is $19.99. "
            "International shipping is available to 50+ countries — rates vary at checkout."
        ),
        "category": "shipping",
    },
    "appointment_booking": {
        "keywords": [
            "appointment", "book", "schedule", "consultation", "meeting", "visit",
            "book appointment", "schedule appointment", "make appointment", "set up meeting",
            "consultation booking", "service appointment", "installation appointment",
            "available times", "available slots", "book a time", "schedule a call",
            "when can i come in", "how to book", "reschedule", "cancel appointment",
        ],
        "policy": (
            "You can book, reschedule, or cancel appointments at our website under 'Book a Service'. "
            "Available slots show in real time — most locations offer same-week appointments. "
            "You'll receive an email confirmation with a reminder 24 hours before."
        ),
        "category": "appointments",
    },
    "human_agent": {
        "keywords": [
            "human", "agent", "person", "representative", "speak to", "talk to",
            "live agent", "real person", "customer service", "call", "phone",
            "speak with someone", "talk to a person", "connect me", "transfer me",
            "escalate", "manager", "supervisor", "not helpful", "frustrated",
            "need more help", "complex issue", "urgent", "emergency",
        ],
        "policy": (
            "I'll connect you with a live agent right away. "
            "You can also reach our team by phone at 1-800-SUPPORT (Mon–Fri, 8 AM–8 PM) "
            "or via email at support@company.com."
        ),
        "category": "escalation",
    },
    "product_inquiry": {
        "keywords": [
            "product", "item", "available", "stock", "inventory", "in stock", "out of stock",
            "product info", "product details", "specifications", "features", "description",
            "color options", "size options", "variants", "model", "version",
            "compatible with", "works with", "warranty", "guarantee", "quality",
            "new product", "popular product", "best seller", "recommend",
        ],
        "policy": (
            "Full product details, specs, and stock availability are on each product page. "
            "Use filters on the catalogue to find your size, color, or model. "
            "If something is out of stock, click 'Notify Me' to get an alert when it returns."
        ),
        "category": "products",
    },
    "out_of_scope": {
        "keywords": [
            "weather", "news", "sports", "recipe", "joke", "entertainment",
            "unrelated", "random", "off topic",
        ],
        "policy": (
            "I'm a customer support assistant, so I'm best placed to help with orders, billing, "
            "accounts, returns, shipping, or appointments. "
            "Is there anything along those lines I can help you with today?"
        ),
        "category": "meta",
    },
}


# ── Semantic encoder (sentence-transformers) ──────────────────────────────────

class _SemanticEncoder:
    """
    Wraps sentence-transformers/all-MiniLM-L6-v2.
    Gracefully disabled if the package is not installed.
    """

    def __init__(self):
        self._model   = None
        self._enabled = False
        self._load()

    def _load(self):
        try:
            from sentence_transformers import SentenceTransformer
            model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
            self._model   = SentenceTransformer(model_name)
            self._enabled = True
            logger.info(f"[SemanticEncoder] loaded: {model_name}")
        except ImportError:
            logger.warning(
                "[SemanticEncoder] sentence-transformers not installed — "
                "semantic layer disabled. Run: pip install sentence-transformers"
            )
        except Exception as e:
            logger.warning(f"[SemanticEncoder] load failed: {e} — semantic layer disabled")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def encode(self, texts: list) -> np.ndarray:
        """Return (N, 384) float32 array. Raises if not enabled."""
        if not self._enabled:
            raise RuntimeError("SemanticEncoder not available")
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# ── Qdrant vector store ───────────────────────────────────────────────────────

class _VectorStore:
    """
    Qdrant local vector store for dense semantic retrieval.

    Persists to QDRANT_PATH (default: ./qdrant_store).
    Falls back gracefully if qdrant-client is not installed.
    """

    def __init__(self, encoder: _SemanticEncoder):
        self._client    = None
        self._encoder   = encoder
        self._enabled   = False
        self._init()

    def _init(self):
        if not self._encoder.enabled:
            logger.info("[VectorStore] skipped — SemanticEncoder unavailable")
            return
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            # Cloud instance takes priority; fall back to local disk store
            if _QDRANT_URL and _QDRANT_KEY:
                self._client = QdrantClient(url=_QDRANT_URL, api_key=_QDRANT_KEY)
                logger.info(f"[VectorStore] connected to cloud Qdrant: {_QDRANT_URL}")
            else:
                self._client = QdrantClient(path=_QDRANT_PATH)
                logger.info(f"[VectorStore] using local Qdrant store: {_QDRANT_PATH}")

            self._enabled = True

            # Create collection if it doesn't exist yet
            existing = [c.name for c in self._client.get_collections().collections]
            if _COLLECTION not in existing:
                self._client.create_collection(
                    collection_name=_COLLECTION,
                    vectors_config=VectorParams(size=_VECTOR_DIM, distance=Distance.COSINE),
                )
                logger.info(f"[VectorStore] created collection '{_COLLECTION}'")
                self._needs_upsert = True
            else:
                # Re-upsert if collection is empty (e.g. after cloud reset)
                info  = self._client.get_collection(_COLLECTION)
                count = (
                    getattr(info, "vectors_count", None) or
                    getattr(info, "points_count",  None) or 0
                )
                self._needs_upsert = (count == 0)
                if self._needs_upsert:
                    logger.warning(f"[VectorStore] '{_COLLECTION}' empty — will re-upsert on next _build_indices")
                else:
                    logger.info(f"[VectorStore] '{_COLLECTION}' has {count} vectors")

        except ImportError:
            logger.warning(
                "[VectorStore] qdrant-client not installed — "
                "semantic layer disabled. Run: pip install qdrant-client"
            )
        except Exception as e:
            logger.warning(f"[VectorStore] init failed: {e} — semantic layer disabled")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def upsert_policies(self, policies: dict):
        """
        Embed and upsert all policies into Qdrant.
        Each policy is stored with its name and policy text as payload.
        Called by KnowledgeBase._build_indices() so it stays in sync.
        """
        if not self._enabled:
            return
        try:
            from qdrant_client.models import PointStruct

            names   = list(policies.keys())
            # Embed: keywords + policy text for richer semantic coverage
            docs    = [
                " ".join(policies[n]["keywords"]) + " " + policies[n]["policy"]
                for n in names
            ]
            vectors = self._encoder.encode(docs)

            points = [
                PointStruct(
                    id      = idx,
                    vector  = vectors[idx].tolist(),
                    payload = {
                        "name":     names[idx],
                        "policy":   policies[names[idx]]["policy"],
                        "category": policies[names[idx]]["category"],
                    },
                )
                for idx in range(len(names))
            ]
            self._client.upsert(collection_name=_COLLECTION, points=points)
            logger.info(f"[VectorStore] upserted {len(points)} policy vectors")
        except Exception as e:
            logger.warning(f"[VectorStore] upsert failed: {e}")

    def search(self, query: str, top_k: int = 5) -> list:
        """
        Semantic search. Returns list of dicts:
          [{"name": ..., "policy": ..., "category": ..., "score": float}, ...]
        """
        if not self._enabled:
            return []
        try:
            q_vec = self._encoder.encode_one(query).tolist()
            # qdrant-client >= 1.9 uses query_points; older uses search
            if hasattr(self._client, "query_points"):
                from qdrant_client.models import Query
                resp    = self._client.query_points(
                    collection_name=_COLLECTION,
                    query=q_vec,
                    limit=top_k,
                    with_payload=True,
                )
                results = resp.points
            else:
                results = self._client.search(
                    collection_name=_COLLECTION,
                    query_vector=q_vec,
                    limit=top_k,
                    with_payload=True,
                )
            return [
                {
                    "name":     r.payload["name"],
                    "policy":   r.payload["policy"],
                    "category": r.payload["category"],
                    "score":    float(r.score),
                }
                for r in results
            ]
        except Exception as e:
            logger.warning(f"[VectorStore] search failed: {e}")
            return []


# ── Knowledge base ────────────────────────────────────────────────────────────

class KnowledgeBase:
    """
    Hybrid knowledge base: TF-IDF keyword layer + Qdrant dense semantic layer.

    Retrieval score = _KW_WEIGHT × keyword_score + _SEM_WEIGHT × semantic_score

    If sentence-transformers or qdrant-client are not installed the semantic
    layer is silently disabled and the system falls back to TF-IDF only
    (same behaviour as before).
    """

    def __init__(self):
        self.policies = POLICIES
        self._stemmer = PorterStemmer()

        # Semantic components — initialised before TF-IDF so upsert happens
        # inside _build_indices() after the TF-IDF matrices are ready
        self._encoder     = _SemanticEncoder()
        self._vector_store = _VectorStore(self._encoder)

        self._build_indices()

    # ── Stemmer ───────────────────────────────────────────────────────────────

    def _stem(self, text: str) -> str:
        return " ".join(self._stemmer.stem(w) for w in text.lower().split())

    # ── Index construction ────────────────────────────────────────────────────

    def _build_indices(self):
        """
        Build TF-IDF matrices and (re)index Qdrant vectors.
        Called at startup and after policy expansion during retrain.
        """
        names         = list(self.policies.keys())
        keyword_docs  = [" ".join(self.policies[n]["keywords"]) for n in names]
        fulltext_docs = [
            " ".join(self.policies[n]["keywords"]) + " " + self.policies[n]["policy"]
            for n in names
        ]

        # ── Layer 1: TF-IDF keyword index ─────────────────────────────────────
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True,
            preprocessor=self._stem,
            strip_accents="unicode",
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(keyword_docs)

        # Fulltext TF-IDF kept for context_awareness_score (used in prompt builder)
        self.fulltext_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            preprocessor=self._stem,
            strip_accents="unicode",
        )
        self.fulltext_matrix = self.fulltext_vectorizer.fit_transform(fulltext_docs)

        # ── Layer 2: Qdrant dense vectors ─────────────────────────────────────
        # Always upsert — keeps vectors in sync with any policy changes
        self._vector_store.upsert_policies(self.policies)

        semantic_status = (
            f"Qdrant ({_QDRANT_PATH})" if self._vector_store.enabled else "disabled (TF-IDF only)"
        )
        logger.info(
            f"KnowledgeBase ready — {len(self.policies)} policies | "
            f"TF-IDF ✓ | Semantic: {semantic_status}"
        )

    # ── Hybrid retrieval ──────────────────────────────────────────────────────

    def retrieve_relevant_policies(self, query: str, top_k: int = 3) -> list:
        """
        Hybrid retrieval: TF-IDF keyword score + Qdrant semantic score.

        Returns list of dicts:
          [{"name", "policy", "category", "relevance"}, ...]
        sorted by combined score descending.
        """
        names = list(self.policies.keys())

        # ── Layer 1: TF-IDF keyword scores ────────────────────────────────────
        try:
            stemmed  = self._stem(query)
            kw_vec   = self.vectorizer.transform([stemmed])
            kw_sims  = cosine_similarity(kw_vec, self.tfidf_matrix)[0]
        except Exception as e:
            logger.error(f"[KnowledgeBase] TF-IDF retrieval error: {e}")
            kw_sims = np.zeros(len(names))

        # ── Layer 2: Qdrant semantic scores ───────────────────────────────────
        sem_scores = np.zeros(len(names))
        if self._vector_store.enabled:
            try:
                sem_results = self._vector_store.search(query, top_k=len(names))
                name_to_idx = {n: i for i, n in enumerate(names)}
                for r in sem_results:
                    idx = name_to_idx.get(r["name"])
                    if idx is not None:
                        sem_scores[idx] = r["score"]
            except Exception as e:
                logger.warning(f"[KnowledgeBase] semantic search error: {e}")

        # ── Combine ───────────────────────────────────────────────────────────
        combined = _KW_WEIGHT * kw_sims + _SEM_WEIGHT * sem_scores
        top_idx  = np.argsort(combined)[::-1][:top_k]

        results = []
        for idx in top_idx:
            score     = float(combined[idx])
            top_score = float(combined[top_idx[0]])
            if score < 0.05:
                break
            if top_score > 0 and score < top_score * 0.25:
                break
            name = names[idx]
            results.append({
                "name":      name,
                "policy":    self.policies[name]["policy"],
                "category":  self.policies[name]["category"],
                "relevance": round(score, 4),
            })

        return results

    # ── Direct lookups ────────────────────────────────────────────────────────

    def get_policy_by_intent(self, intent: str):
        p = self.policies.get(intent)
        return p["policy"] if p else None

    def get_context_awareness_score(self, query: str, intent: str) -> float:
        try:
            policy = self.policies.get(intent)
            if not policy:
                return 0.0
            stemmed_q = self._stem(query)
            stemmed_p = self._stem(" ".join(policy["keywords"]))
            q_vec     = self.vectorizer.transform([stemmed_q])
            p_vec     = self.vectorizer.transform([stemmed_p])
            return float(cosine_similarity(q_vec, p_vec)[0][0])
        except Exception as e:
            logger.error(f"[KnowledgeBase] context awareness error: {e}")
            return 0.0

    # ── Semantic-only search (used by oos_router Stage 0) ─────────────────────

    def semantic_search(self, query: str, top_k: int = 1) -> list:
        """
        Pure dense vector search — bypasses TF-IDF.
        Used by oos_router for the fast semantic pre-filter.
        Returns same format as retrieve_relevant_policies.
        """
        if not self._vector_store.enabled:
            return self.retrieve_relevant_policies(query, top_k=top_k)
        results = self._vector_store.search(query, top_k=top_k)
        return [
            {
                "name":      r["name"],
                "policy":    r["policy"],
                "category":  r["category"],
                "relevance": round(r["score"], 4),
            }
            for r in results
        ]


# ── Contextual RAG pipeline ───────────────────────────────────────────────────

class ContextualRAG:
    """RAG pipeline: hybrid retrieval → augmented prompt for Mistral."""

    def __init__(self):
        self.knowledge_base    = KnowledgeBase()
        self.retrieval_history = []

    def generate_context_prompt(self, query, intent, confidence, retrieved_policies):
        ctx_score   = self.knowledge_base.get_context_awareness_score(query, intent)
        ctx_section = self._format_context(retrieved_policies)

        return (
            "[SYSTEM]\n"
            "You are a concise customer support AI. "
            "STRICT RULES: respond in exactly 1 sentence, ≤ 30 words. "
            "Do NOT start with 'Sure', 'Of course', 'Certainly', 'Great question', etc. "
            "Use ONLY the CONTEXT below. Do NOT invent facts.\n\n"
            "[CONTEXT_ANALYSIS]\n"
            f"Query: \"{query}\"\n"
            f"Intent: {intent} ({confidence:.0%} confidence)\n"
            f"Relevance: {ctx_score:.0%}\n\n"
            "[KNOWLEDGE BASE CONTEXT]\n"
            f"{ctx_section}\n\n"
            "[RESPONSE RULES]\n"
            "1. Pick the most relevant policy fact\n"
            "2. Answer directly — no bullet points, no lists\n"
            "3. If not covered: 'I don't have that detail — please contact our support team.'\n"
            "4. Professional, empathetic, conversational tone\n"
            "5. ≤ 30 words. 1 sentence. No preamble.\n\n"
            f"[CUSTOMER MESSAGE] {query}\n\n"
            "[RESPONSE — 1 sentence, ≤ 30 words, no preamble]:"
        )

    def _format_context(self, retrieved_policies):
        if not retrieved_policies:
            return "No matching policy. Use general support fallback."
        p         = retrieved_policies[0]
        indicator = (
            "HIGH"     if p["relevance"] >= 0.6 else
            "MODERATE" if p["relevance"] >= 0.3 else
            "LOW"
        )
        return f"[{indicator} RELEVANCE — {p['name']}]\n{p['policy']}"

    def retrieve_and_augment(self, query: str, intent: str, confidence: float):
        retrieved    = self.knowledge_base.retrieve_relevant_policies(query, top_k=3)
        primary_text = self.knowledge_base.get_policy_by_intent(intent)

        if primary_text and (confidence < 0.3 or confidence > 0.8):
            primary = {
                "name":      intent,
                "policy":    primary_text,
                "category":  "primary",
                "relevance": 1.0 if confidence > 0.8 else 0.5,
            }
            if not any(p["name"] == intent for p in retrieved):
                retrieved = [primary] + retrieved[:2]

        prompt = self.generate_context_prompt(query, intent, confidence, retrieved)

        self.retrieval_history.append({
            "query":           query,
            "intent":          intent,
            "retrieved_count": len(retrieved),
            "top_relevance":   retrieved[0]["relevance"] if retrieved else 0,
        })

        return {
            "prompt":             prompt,
            "retrieved_policies": retrieved,
            "context_score":      confidence,
            "retrieval_metadata": {"query": query, "intent": intent},
        }

    def get_retrieval_stats(self):
        if not self.retrieval_history:
            return {}
        return {
            "total_queries":     len(self.retrieval_history),
            "avg_relevance":     np.mean([r["top_relevance"] for r in self.retrieval_history]),
            "semantic_enabled":  self.knowledge_base._vector_store.enabled,
            "retrieval_history": self.retrieval_history[-10:],
        }
