"""
app.py — PROD Support Chatbot
Pipeline: Intent Classifier → Semantic KB search → Mistral (primary answer)
OOS:      Gemini (contextual analysis) → SerpAPI (web) | Mistral (knowledge)
Live:     Socket.IO agent dashboard
"""
import os, logging, uuid, threading, json, re, random
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit, join_room
from flask_cors import CORS
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "prod-support-secret-key")
CORS(app, origins=os.getenv("ALLOWED_ORIGINS", "*"))
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Config from env ───────────────────────────────────────────────────────────
MODEL_PATH                  = os.getenv("MODEL_PATH", "intent_model_v2_1777554249.pkl")
LLM_TEMPERATURE             = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS              = int(os.getenv("LLM_MAX_TOKENS", "120"))
MAX_RESPONSE_WORDS          = int(os.getenv("MAX_RESPONSE_WORDS", "80"))
TEMPLATE_CONFIDENCE         = float(os.getenv("TEMPLATE_CONFIDENCE", "0.75"))
ESCALATION_TURN_THRESHOLD   = int(os.getenv("ESCALATION_TURN_THRESHOLD", "5"))
GEMINI_API_KEY              = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL                = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ── Intent classifier ─────────────────────────────────────────────────────────
from intent_model import IntentClassifier
classifier = IntentClassifier()
classifier.load_model(MODEL_PATH)
log.info(f"Classifier loaded: {MODEL_PATH}")

# ── RAG engine ────────────────────────────────────────────────────────────────
from rag_system import ContextualRAG
rag_engine = ContextualRAG()
log.info("RAG engine ready")

# ── OOS router (SerpAPI → Gemini → Mistral) ───────────────────────────────────
from oos_router import get_router
oos_router = get_router(rag_engine=rag_engine)
log.info("OOS router ready")

# ── LLM clients ───────────────────────────────────────────────────────────────
from llm_clients import get_mistral
mistral         = get_mistral()
MISTRAL_ENABLED = mistral.enabled

# Gemini — used for OOS analysis only (not primary answer writer)
lc_gemini    = None
gemini_model = None

if GEMINI_API_KEY:
    # Try primary model, fall back to GEMINI_FALLBACK_MODEL if overloaded
    _gemini_models = [
        GEMINI_MODEL,
        os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.0-flash"),
    ]
    for _gm in dict.fromkeys(_gemini_models):  # deduplicate, preserve order
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            lc_gemini = ChatGoogleGenerativeAI(
                model=_gm,
                google_api_key=GEMINI_API_KEY,
                temperature=0.1,
                convert_system_message_to_human=True,
                max_retries=2,
            )
            log.info(f"Gemini (LangChain) ready — {_gm}")
            break
        except Exception as e:
            log.warning(f"Gemini LangChain init failed for {_gm}: {e}")

    for _gm in dict.fromkeys(_gemini_models):
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel(
                model_name=_gm,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 200,
                    "top_p": 0.9,
                },
            )
            log.info(f"Gemini (SDK) ready — {_gm}")
            break
        except Exception as e:
            log.warning(f"Gemini SDK init failed for {_gm}: {e}")
else:
    log.warning("Gemini disabled — GEMINI_API_KEY not set")

LLM_ENABLED = MISTRAL_ENABLED or bool(GEMINI_API_KEY)

# ── Session state ─────────────────────────────────────────────────────────────
live_sessions:     dict = {}
_bot_turn_tracker: dict = {}
_turn_lock               = threading.Lock()

# ── Support scope string ──────────────────────────────────────────────────────
SUPPORT_TOPICS = (
    "orders, returns, refunds, billing, payments, shipping, "
    "account help, technical support, or appointments"
)

# ── Preamble stripper ─────────────────────────────────────────────────────────
_PREAMBLE_RE = re.compile(
    r"^(of course[,!]?\s*|sure[,!]?\s*|certainly[,!]?\s*|absolutely[,!]?\s*"
    r"|great question[,!]?\s*|happy to help[,!]?\s*|i.?d be happy to[,!]?\s*"
    r"|no problem[,!]?\s*|thanks for (reaching out|contacting us)[,!]?\s*"
    r"|i understand[,!]?\s*|i see[,!]?\s*)",
    re.IGNORECASE,
)

def clean_response(text: str) -> str:
    if not text:
        return text
    text = _PREAMBLE_RE.sub("", text).strip()
    if text:
        text = text[0].upper() + text[1:]
    words = text.split()
    if len(words) > MAX_RESPONSE_WORDS:
        truncated = " ".join(words[:MAX_RESPONSE_WORDS])
        last_end  = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
        text      = truncated[:last_end + 1] if last_end > 20 else truncated + "..."
    return re.sub(r"\n{3,}", "\n\n", text).strip()

# ── Social intent handlers ────────────────────────────────────────────────────
def _handle_greeting(msg: str) -> str:
    m = msg.lower()
    if any(w in m for w in ["good morning", "morning"]):
        return "Good morning! How can I help you today?"
    if any(w in m for w in ["good afternoon", "afternoon"]):
        return "Good afternoon! What can I help you with?"
    if any(w in m for w in ["good evening", "evening", "good night"]):
        return "Good evening! What can I help you with?"
    if any(w in m for w in ["how are you", "how r you", "how are u", "you ok"]):
        return "Doing great, thanks for asking! What can I help you with today?"
    if any(w in m for w in ["who are you", "what are you", "are you a bot", "are you human", "are you ai"]):
        return (
            "I'm an AI-powered customer support assistant. "
            "I can help with orders, returns, billing, shipping, account issues, and more. "
            "What do you need?"
        )
    return random.choice([
        "Hello! How can I help you today?",
        "Hi there! What can I assist you with?",
        "Hey! What do you need help with today?",
        "Hi! I'm here to help — what's going on?",
    ])

def _handle_goodbye(msg: str) -> str:
    return random.choice([
        "Goodbye! Feel free to reach out anytime. Have a great day!",
        "Take care! Come back if you need anything.",
        "Bye! It was a pleasure helping you.",
        "All the best! Don't hesitate to reach out if you need help.",
    ])

def _handle_thanks(msg: str) -> str:
    return random.choice([
        "You're welcome! Anything else I can help with?",
        "Happy to help! Let me know if you need anything else.",
        "Glad I could assist! Feel free to ask if you have more questions.",
        "No problem at all! Is there anything else?",
    ])

def _handle_feedback(msg: str) -> str:
    return random.choice([
        "Thank you for your feedback — we really appreciate it! Is there anything else I can help with?",
        "Thanks for sharing that — your input helps us improve. Anything else I can do for you?",
        "We appreciate the feedback! Is there anything else you need help with today?",
    ])

# ── Fast-path template cache (zero LLM latency) ───────────────────────────────
_FAST_TEMPLATES = {
    "order_status":        "Track your order in 'My Orders' using your order number or email. Standard delivery is 5–7 days; expedited is 2–3 days.",
    "return_refund":       "Returns are accepted within 30 days (unused, original packaging). Go to 'My Orders' → select the item → 'Return/Exchange'.",
    "cancellation":        "Orders can be cancelled within 2 hours via 'My Orders' → 'Cancel Order'. If it's already shipped, start a return instead.",
    "billing_issue":       "Share your order number or email and we'll investigate — duplicate charges are refunded within 1 business day.",
    "shipping_info":       "Standard shipping (5–7 days) is free on orders over $35. Expedited is $9.99 (2–3 days), overnight is $19.99.",
    "account_help":        "Click 'Forgot Password' on the login page to reset. Locked accounts unlock automatically after 15 minutes.",
    "technical_support":   "Try clearing your cache, restarting the app, or using a different browser. For login issues, try incognito mode.",
    "appointment_booking": "Book or reschedule at our website under 'Book a Service'. You'll receive a confirmation email right away.",
    "product_inquiry":     "Full details and stock availability are on each product page. Click 'Notify Me' for out-of-stock items.",
    "human_agent":         "Connecting you now. You can also call 1-800-SUPPORT (Mon–Fri 8AM–8PM) or email support@company.com.",
    "greeting":  None,
    "goodbye":   None,
    "thanks":    None,
    "feedback":  None,
}

# ── Mistral direct answer (in-scope fast path) ────────────────────────────────
def _mistral_direct(user_message: str, policy_ctx: str, turn_number: int = 1) -> Optional[str]:
    if not MISTRAL_ENABLED:
        return None
    try:
        escalation = (
            " If the issue isn't resolved, offer to connect them with a live agent."
            if turn_number >= 4 else ""
        )
        system = (
            "You are a concise customer support assistant. "
            "Answer in 1–2 sentences (max 35 words) using ONLY the policy below. "
            "Be direct and helpful. No preamble. No filler."
            + escalation
        )
        result = mistral.chat(
            [
                {"role": "system", "content": system},
                {"role": "user",   "content": f"POLICY:\n{policy_ctx}\n\nCUSTOMER: {user_message}\n\nAnswer:"},
            ],
            temperature=0.15,
            max_tokens=80,
        )
        if result:
            log.info(f"[Mistral] direct answer: {len(result.split())} words")
            return result
    except Exception as e:
        log.warning(f"[Mistral] direct failed: {e}")
    return None

# ── Fallback LLM caller (human_agent path) ────────────────────────────────────
def call_llm(prompt: str) -> Optional[str]:
    """Mistral → Gemini LangChain → Gemini SDK."""
    if MISTRAL_ENABLED:
        try:
            resp = mistral.generate_content(prompt)
            if resp and resp.text:
                return resp.text
        except Exception as e:
            log.warning(f"call_llm Mistral failed: {e}")
    if lc_gemini:
        try:
            from langchain_core.messages import HumanMessage
            resp = lc_gemini.invoke([HumanMessage(content=prompt)])
            text = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
            if text:
                return text
        except Exception as e:
            log.warning(f"call_llm Gemini LC failed: {e}")
    if gemini_model:
        try:
            resp = gemini_model.generate_content(prompt)
            return resp.text.strip()
        except Exception as e:
            log.warning(f"call_llm Gemini SDK failed: {e}")
    return None

# ── Turn tracker ──────────────────────────────────────────────────────────────
def _session_key(req) -> str:
    return f"{req.remote_addr}|{req.headers.get('User-Agent','')[:60]}"

def _increment_turn(key: str, resolved: bool) -> dict:
    with _turn_lock:
        s = _bot_turn_tracker.setdefault(key, {"turns": 0, "unresolved": 0})
        s["turns"] += 1
        s["unresolved"] = 0 if resolved else s["unresolved"] + 1
        return dict(s)

def _should_escalate(state: dict) -> bool:
    return state["unresolved"] >= ESCALATION_TURN_THRESHOLD

def _ts() -> str:
    return datetime.now().strftime("%H:%M")

# ── Main response generator ───────────────────────────────────────────────────
def generate_bot_response(intent: str, user_message: str,
                           confidence: float, turn_number: int = 1) -> str:

    # ── Social intents ────────────────────────────────────────────────────────
    if intent == "greeting": return _handle_greeting(user_message)
    if intent == "goodbye":  return _handle_goodbye(user_message)
    if intent == "thanks":   return _handle_thanks(user_message)
    if intent == "feedback": return _handle_feedback(user_message)

    # ── High-confidence template (zero latency) ───────────────────────────────
    if (intent in _FAST_TEMPLATES
            and _FAST_TEMPLATES[intent] is not None
            and confidence >= TEMPLATE_CONFIDENCE):
        log.info(f"[Template] intent={intent} conf={confidence:.2f}")
        return _FAST_TEMPLATES[intent]

    # ── OOS / low-confidence → OOS router ────────────────────────────────────
    if intent == "out_of_scope" or confidence < 0.30:
        response, oos_type = oos_router.route(
            query=user_message, intent=intent, confidence=confidence,
        )
        log.info(f"[OOS] type={oos_type} conf={confidence:.2f}")
        return clean_response(response)

    # ── Human agent escalation ────────────────────────────────────────────────
    if intent == "human_agent":
        policy = rag_engine.knowledge_base.get_policy_by_intent("human_agent")
        if policy and LLM_ENABLED:
            r = call_llm(
                f"Customer wants a human agent. In ONE sentence, acknowledge and give the contact option.\n\n"
                f"Policy:\n{policy}\n\nCustomer: {user_message}\n\nResponse:"
            )
            if r:
                return clean_response(r)
        return policy or f"I'm your support assistant. I can help with {SUPPORT_TOPICS}."

    # ── In-scope: RAG → Mistral ───────────────────────────────────────────────
    try:
        rag_result = rag_engine.retrieve_and_augment(user_message, intent, confidence)
        retrieved  = rag_result["retrieved_policies"]

        if retrieved:
            top        = retrieved[0]
            policy_ctx = f"[{top['name']} — {top['relevance']:.0%} relevance]\n{top['policy']}"
        else:
            policy_ctx = rag_engine.knowledge_base.get_policy_by_intent(intent) or ""

        if policy_ctx and LLM_ENABLED:
            answer = _mistral_direct(user_message, policy_ctx, turn_number)
            if answer:
                return clean_response(answer)

        # Fallback: return raw policy text
        if policy_ctx and confidence >= 0.35:
            return policy_ctx

        if retrieved and retrieved[0]["relevance"] >= 0.1:
            return retrieved[0]["policy"]

        return f"I don't have enough detail to answer that confidently. I can help with {SUPPORT_TOPICS}."

    except Exception as e:
        log.error(f"generate_bot_response error: {e}")
        return "I'm here to help — please let me know what you need."

# ── HTTP routes ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index_responsive.html")

@app.route("/agent")
def agent():
    return render_template("agent_dashboard.html")

@app.route("/admin")
def admin():
    return render_template("admin_retrain.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data         = request.get_json(force=True) or {}
        user_message = (data.get("message") or data.get("text") or "").strip()
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        result = classifier.predict(user_message)
        intent, confidence = result if isinstance(result, tuple) else (
            result.get("intent", "unknown"), float(result.get("confidence", 0))
        )

        key        = _session_key(request)
        resolved   = confidence >= 0.50 and intent not in ("out_of_scope", "unknown")
        turn_state = _increment_turn(key, resolved)
        turn_num   = turn_state["turns"]

        response = generate_bot_response(intent, user_message, confidence, turn_num)

        # Escalation only for in-scope unresolved turns
        escalate = _should_escalate(turn_state) and intent not in ("out_of_scope", "unknown")
        if escalate:
            response += "\n\nThis issue may need more personalised help. Would you like me to connect you with a live agent?"

        # Path label for debugging
        if intent in _FAST_TEMPLATES and _FAST_TEMPLATES[intent] and confidence >= TEMPLATE_CONFIDENCE:
            path = "template"
        elif intent in ("greeting", "goodbye", "thanks", "feedback"):
            path = "social"
        elif intent == "out_of_scope" or confidence < 0.30:
            path = "oos"
        else:
            path = "mistral"

        log.info(f"[Chat] intent={intent} conf={confidence:.2f} turn={turn_num} path={path} words={len(response.split())}")

        return jsonify({
            "response":   response,
            "intent":     intent,
            "confidence": round(confidence, 3),
            "turn":       turn_num,
            "escalate":   escalate,
            "path":       path,
        })

    except Exception as e:
        log.error(f"[Chat] endpoint error: {e}")
        return jsonify({"error": "Internal server error", "response": "Something went wrong. Please try again."}), 500

@app.route("/health", methods=["GET"])
def health():
    kb = rag_engine.knowledge_base
    return jsonify({
        "status":               "healthy",
        "pipeline":             "Intent → Semantic KB → Mistral | OOS: Gemini → SerpAPI/Mistral",
        "primary_llm":          "mistral",
        "mistral_enabled":      MISTRAL_ENABLED,
        "mistral_model":        os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        "gemini_analyser":      lc_gemini is not None,
        "gemini_model":         GEMINI_MODEL,
        "serpapi_enabled":      bool(os.getenv("SERPAPI_API_KEY", "") or os.getenv("SERPAPI_KEY", "")),
        "llm_enabled":          LLM_ENABLED,
        "qdrant_mode":          "cloud" if (os.getenv("QDRANT_URL") and os.getenv("QDRANT_API_KEY")) else "local",
        "qdrant_semantic":      kb._vector_store.enabled,
        "semantic_encoder":     kb._encoder.enabled,
        "template_confidence":  TEMPLATE_CONFIDENCE,
        "escalation_threshold": ESCALATION_TURN_THRESHOLD,
        "max_response_words":   MAX_RESPONSE_WORDS,
        "model":                MODEL_PATH,
    })

# ── Admin routes ──────────────────────────────────────────────────────────────
@app.route("/admin/retrain", methods=["POST"])
def admin_retrain():
    try:
        data  = request.get_json(force=True) or {}
        force = data.get("force", False)
        def _run():
            from retrain_from_conversations import run_retrain
            result = run_retrain(
                classifier=classifier,
                rag_engine=rag_engine,
                llm_model=gemini_model,
                force=force,
            )
            log.info(f"[Retrain] complete: {result}")
        threading.Thread(target=_run, daemon=True, name="manual-retrain").start()
        return jsonify({"status": "retrain started"}), 202
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/admin/retrain/status", methods=["GET"])
def admin_retrain_status():
    """Return current auto-retrain configuration and state."""
    try:
        from conversation_logger import (
            _AUTO_RETRAIN_ENABLED, _AUTO_RETRAIN_THRESHOLD,
            _retrain_in_progress, _last_retrain_count, _count_total_pairs,
        )
        total   = _count_total_pairs()
        pending = max(0, total - _last_retrain_count)
        return jsonify({
            "auto_retrain_enabled":   _AUTO_RETRAIN_ENABLED,
            "threshold":              _AUTO_RETRAIN_THRESHOLD,
            "total_pairs_logged":     total,
            "pairs_since_last_train": pending,
            "retrain_in_progress":    _retrain_in_progress,
            "next_retrain_in":        max(0, _AUTO_RETRAIN_THRESHOLD - pending),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/admin/evaluate", methods=["POST"])
def admin_evaluate():
    try:
        def _run():
            from rag_evaluator import run_full_evaluation
            run_full_evaluation(rag_engine, classifier, llm_model=gemini_model)
        threading.Thread(target=_run, daemon=True).start()
        return jsonify({"status": "evaluation started"}), 202
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/admin/ood", methods=["POST"])
def admin_ood():
    try:
        def _run():
            from synthetic_conversation_generator import run as run_ood
            result = run_ood(save_training_csv=True)
            log.info(f"OOD complete: {result.get('ragas_samples', 0)} samples")
        threading.Thread(target=_run, daemon=True).start()
        return jsonify({"status": "ood generation started"}), 202
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/admin/qdrant/status", methods=["GET"])
def admin_qdrant_status():
    """Live status of the Qdrant cloud vector store and semantic layer."""
    try:
        kb  = rag_engine.knowledge_base
        vs  = kb._vector_store
        enc = kb._encoder

        status = {
            "semantic_encoder": {
                "enabled": enc.enabled,
                "model":   os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"),
            },
            "vector_store": {
                "enabled":    vs.enabled,
                "mode":       "cloud" if (os.getenv("QDRANT_URL") and os.getenv("QDRANT_API_KEY")) else "local",
                "url":        os.getenv("QDRANT_URL", "local"),
                "collection": "kb_policies",
            },
            "hybrid_weights": {
                "keyword":  float(os.getenv("HYBRID_KEYWORD_WEIGHT",  "0.5")),
                "semantic": float(os.getenv("HYBRID_SEMANTIC_WEIGHT", "0.5")),
            },
        }

        # Live collection stats from Qdrant
        if vs.enabled and vs._client:
            try:
                info = vs._client.get_collection("kb_policies")
                # vectors_count moved to points_count in newer qdrant-client versions
                count = (
                    getattr(info, "vectors_count", None) or
                    getattr(info, "points_count",  None) or
                    getattr(getattr(info, "result", None), "vectors_count", None) or 0
                )
                status["vector_store"]["vectors_count"] = count
                status["vector_store"]["collection_status"] = str(info.status)
            except Exception as e:
                status["vector_store"]["collection_error"] = str(e)

        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/qdrant/reindex", methods=["POST"])
def admin_qdrant_reindex():
    """Force re-embed and re-upsert all KB policies into Qdrant."""
    try:
        def _run():
            kb = rag_engine.knowledge_base
            kb._vector_store.upsert_policies(kb.policies)
            log.info("[Qdrant] Manual reindex complete")
        threading.Thread(target=_run, daemon=True, name="qdrant-reindex").start()
        return jsonify({"status": "reindex started"}), 202
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/qdrant/search", methods=["POST"])
def admin_qdrant_search():
    """Test semantic search against Qdrant. Body: {\"query\": \"...\"}"""
    try:
        data  = request.get_json(force=True) or {}
        query = (data.get("query") or "").strip()
        if not query:
            return jsonify({"error": "No query provided"}), 400
        kb       = rag_engine.knowledge_base
        hybrid   = kb.retrieve_relevant_policies(query, top_k=3)
        semantic = kb.semantic_search(query, top_k=3)
        return jsonify({
            "query":            query,
            "hybrid":           hybrid,
            "semantic":         semantic,
            "semantic_enabled": kb._vector_store.enabled,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/ood/results", methods=["GET"])
def admin_ood_results():
    try:
        from pathlib import Path
        scores_path  = Path("eval_results/ood_ragas_scores.json")
        dataset_path = Path("eval_results/ood_ragas_dataset.json")
        if not scores_path.exists():
            return jsonify({"status": "not_run", "message": "No OOD evaluation run yet"})
        with open(scores_path) as f:
            scores = json.load(f)
        sample_count = 0
        if dataset_path.exists():
            with open(dataset_path) as f:
                sample_count = len(json.load(f))
        return jsonify({"status": "ok", "scores": scores, "sample_count": sample_count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/admin/scores", methods=["GET"])
def admin_scores():
    try:
        from rag_evaluator import load_latest_scores, load_benchmark_history
        scores  = load_latest_scores()
        history = load_benchmark_history()
        return jsonify({
            "latest_scores":     scores,
            "benchmark_count":   len(history),
            "benchmark_history": history[-5:],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/admin/conversations", methods=["GET"])
def admin_conversations():
    try:
        from conversation_logger import load_conversations
        from collections import Counter
        convs    = load_conversations()
        sessions = Counter(c["session_id"] for c in convs)
        return jsonify({
            "total_pairs":    len(convs),
            "total_sessions": len(sessions),
            "sessions": [{"id": sid, "turns": cnt} for sid, cnt in sessions.most_common(20)],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Socket.IO ─────────────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    log.info(f"Socket connected: {request.sid}")

@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    for sess_id, s in list(live_sessions.items()):
        if s.get("user_sid") == sid and s["status"] == "active":
            s["status"] = "closed"
            if s.get("agent_sid"):
                emit("user_disconnected", {"session_id": sess_id}, room=s["agent_sid"])
    log.info(f"Socket disconnected: {sid}")

@socketio.on("request_live_chat")
def on_request_live_chat(data):
    sess_id = str(uuid.uuid4())[:8]
    live_sessions[sess_id] = {
        "status":    "queued",
        "user_sid":  request.sid,
        "agent_sid": None,
        "history":   data.get("history", []),
        "started":   datetime.now(),
    }
    join_room(sess_id)
    queued_count = len([s for s in live_sessions.values() if s["status"] == "queued"])
    emit("queued", {"session_id": sess_id, "position": queued_count})
    socketio.emit("new_session", {
        "session_id": sess_id,
        "started":    _ts(),
        "preview":    (data.get("history") or [{}])[-1].get("text", "")[:80],
    }, room="agents")
    log.info(f"Live chat queued: {sess_id}")

@socketio.on("user_message")
def on_user_message(data):
    sess_id = data.get("session_id")
    text    = (data.get("text") or "").strip()
    if not sess_id or not text or sess_id not in live_sessions:
        return
    msg = {"from": "user", "text": text, "time": _ts()}
    live_sessions[sess_id]["history"].append(msg)
    socketio.emit("agent_receive", {"session_id": sess_id, "message": msg}, room="agents")

@socketio.on("agent_join")
def on_agent_join():
    join_room("agents")
    queue = [
        {
            "session_id": sid,
            "started":    s["started"].strftime("%H:%M"),
            "status":     s["status"],
            "preview":    s["history"][-1]["text"][:80] if s["history"] else "",
        }
        for sid, s in live_sessions.items()
        if s["status"] != "closed"
    ]
    emit("session_queue", {"sessions": queue})
    log.info(f"Agent joined: {request.sid}")

@socketio.on("agent_accept")
def on_agent_accept(data):
    sess_id = data.get("session_id")
    if sess_id not in live_sessions:
        return
    s              = live_sessions[sess_id]
    s["status"]    = "active"
    s["agent_sid"] = request.sid
    join_room(sess_id)
    socketio.emit("agent_connected", {
        "session_id": sess_id,
        "message":    "You're now connected to a live agent. How can I help you?",
    }, room=sess_id)
    emit("session_history", {"session_id": sess_id, "history": s["history"]})
    socketio.emit("session_updated", {"session_id": sess_id, "status": "active"}, room="agents")
    log.info(f"Agent {request.sid} accepted session {sess_id}")

@socketio.on("agent_message")
def on_agent_message(data):
    sess_id = data.get("session_id")
    text    = (data.get("text") or "").strip()
    if not sess_id or not text or sess_id not in live_sessions:
        return
    msg = {"from": "agent", "text": text, "time": _ts()}
    live_sessions[sess_id]["history"].append(msg)
    socketio.emit("agent_reply", {"session_id": sess_id, "message": msg}, room=sess_id)

@socketio.on("agent_close")
def on_agent_close(data):
    sess_id = data.get("session_id")
    if sess_id not in live_sessions:
        return
    session           = live_sessions[sess_id]
    session["status"] = "closed"
    try:
        from conversation_logger import log_session
        pairs = log_session(
            session_id=sess_id,
            history=session.get("history", []),
            started=session.get("started", datetime.now()),
            channel="live_chat",
        )
        log.info(f"Session {sess_id} logged: {pairs} pairs")
    except Exception as e:
        log.error(f"Failed to log session {sess_id}: {e}")
    socketio.emit("chat_closed", {
        "session_id": sess_id,
        "message":    "The agent has closed this chat. Thank you for contacting us!",
    }, room=sess_id)
    socketio.emit("session_updated", {"session_id": sess_id, "status": "closed"}, room="agents")
    log.info(f"Session closed: {sess_id}")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.getenv("PORT", 8000))
    debug = os.getenv("FLASK_ENV") == "development"
    log.info(f"Starting PROD Support on port {port} (debug={debug})")
    socketio.run(app, host="0.0.0.0", port=port, debug=debug)
