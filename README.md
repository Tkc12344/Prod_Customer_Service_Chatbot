# Customer Support Chatbot

# AI-Powered Customer Support Chatbot

An intelligent customer support platform built with Flask that combines machine learning, Retrieval-Augmented Generation (RAG), and multi-LLM orchestration to deliver accurate, context-aware customer assistance. The system features intent classification, hybrid knowledge retrieval, real-time live agent escalation, automated model retraining, and comprehensive RAG performance evaluation.

Designed for production environments, the chatbot leverages both keyword-based and semantic search to provide reliable answers from internal knowledge bases while seamlessly handling out-of-scope queries through web-enhanced retrieval and advanced language model reasoning.

### Key Capabilities

* **Advanced Intent Classification** using TF-IDF feature engineering and calibrated Support Vector Machines (SVMs)
* **Hybrid Retrieval-Augmented Generation (RAG)** combining TF-IDF keyword search with Qdrant vector similarity search
* **Multi-LLM Architecture** utilizing Mistral for response generation and Gemini for contextual query analysis
* **Intelligent Out-of-Scope (OOS) Handling** with multi-stage routing, semantic analysis, and web-based retrieval
* **Real-Time Live Agent Handoff** through Socket.IO-powered communication channels
* **Continuous Learning** via automatic retraining from newly collected customer interactions
* **RAGAS-Based Evaluation Framework** for monitoring and benchmarking response quality
* **Administrative Dashboard** for model management, evaluation, analytics, and knowledge base monitoring

### Technical Highlights

* Flask 3.1 and Flask-SocketIO backend architecture
* Hybrid search pipeline (TF-IDF + Dense Vector Search)
* Qdrant vector database integration
* LangChain-powered LLM orchestration
* Automatic model fallback and resilience mechanisms
* Synthetic data generation for OOD (Out-of-Distribution) testing
* Real-time conversation logging and analytics
* Scalable and modular microservice-friendly design

The platform is engineered to provide fast, accurate, and reliable customer support while maintaining the flexibility to integrate additional knowledge sources, language models, and enterprise workflows.


## Architecture

```
User message
    │
    ▼
IntentClassifier (TF-IDF + LinearSVC)
    │
    ├── Social intent (greeting / thanks / goodbye / feedback)
    │       └── Static response
    │
    ├── High-confidence in-scope → Template cache (zero latency)
    │
    ├── In-scope (medium confidence)
    │       └── KnowledgeBase hybrid retrieval (TF-IDF + Qdrant)
    │               └── Mistral (primary answer writer)
    │
    └── Out-of-scope / low confidence
            └── OOS Router
                    ├── Stage 0: Semantic KB fast-path
                    ├── Stage 1: Gemini contextual analysis
                    ├── Stage 2: SerpAPI web RAG retrieval
                    └── Stage 3: Mistral grounded answer
```

## Stack

| Component | Technology |
|-----------|-----------|
| Web server | Flask 3.1 + Flask-SocketIO |
| Intent classifier | TF-IDF (word + char n-grams) + LinearSVC (calibrated) |
| Knowledge base | Hybrid: TF-IDF keyword + Qdrant dense vectors (all-MiniLM-L6-v2) |
| Primary LLM | Mistral (`mistral-small-latest`) with automatic fallback to `open-mistral-7b` |
| OOS analyser | Gemini via LangChain (`gemini-1.5-flash` / `gemini-2.0-flash`) |
| Web search | SerpAPI (primary) → LangChain SerpAPI wrapper → DuckDuckGo (fallback) |
| Vector DB | Qdrant Cloud (primary) or local disk store (fallback) |
| Evaluation | RAGAS framework |

## Features

- **Intent classification** — 14 intent classes, TF-IDF word + character n-gram features, calibrated SVM
- **Hybrid RAG** — keyword TF-IDF + Qdrant semantic search with configurable blend weights
- **Multi-LLM pipeline** — Mistral as primary writer, Gemini for OOS analysis, automatic model fallback on rate limits
- **OOS routing** — 7-type contextual classification (live_data, general_knowledge, chitchat, ambiguous, etc.)
- **Live agent handoff** — real-time Socket.IO agent dashboard with session queue
- **Auto-retrain** — automatically retrains intent model after N new live conversation pairs
- **RAGAS evaluation** — full RAG quality benchmarking with benchmark history
- **Admin panel** — retrain, evaluate, OOD synthetic data generation, Qdrant status

## Setup

```bash
# 1. Clone and create virtualenv
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your API keys (only MISTRAL_API_KEY is required)

# 4. Train the intent model
python intent_model.py
# This prints the new model filename — update MODEL_PATH in .env

# 5. Run the app
python app.py
```

Open `http://localhost:8000` in your browser.

## API Keys

| Key | Required | Where to get it |
|-----|----------|-----------------|
| `MISTRAL_API_KEY` | **Required** | [console.mistral.ai](https://console.mistral.ai/) |
| `GEMINI_API_KEY` | Recommended | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| `SERPAPI_KEY` | Optional | [serpapi.com](https://serpapi.com/) — enables live web search |
| `QDRANT_URL` + `QDRANT_API_KEY` | Optional | [cloud.qdrant.io](https://cloud.qdrant.io/) — falls back to local disk |

## Routes

| URL | Method | Description |
|-----|--------|-------------|
| `/` | GET | Customer chat UI |
| `/agent` | GET | Live agent dashboard |
| `/admin` | GET | Admin panel |
| `/chat` | POST | Bot chat API — `{"message": "..."}` |
| `/health` | GET | Health check + component status |
| `/admin/retrain` | POST | Trigger intent model retrain |
| `/admin/retrain/status` | GET | Auto-retrain status and counts |
| `/admin/evaluate` | POST | Run RAGAS evaluation |
| `/admin/ood` | POST | Generate OOD synthetic data |
| `/admin/scores` | GET | Latest RAGAS scores + benchmark history |
| `/admin/conversations` | GET | Live conversation log summary |
| `/admin/qdrant/status` | GET | Qdrant vector store status |
| `/admin/qdrant/reindex` | POST | Force re-embed all KB policies |
| `/admin/qdrant/search` | POST | Test semantic search — `{"query": "..."}` |

## Files

| File | Purpose |
|------|---------|
| `app.py` | Main Flask app — routes, Socket.IO, response pipeline |
| `intent_model.py` | Intent classifier — train and predict (run directly to retrain) |
| `rag_system.py` | Hybrid knowledge base + RAG pipeline (TF-IDF + Qdrant) |
| `llm_clients.py` | Mistral LLM client with automatic fallback |
| `oos_router.py` | Out-of-scope 4-stage contextual pipeline |
| `conversation_logger.py` | Live chat session persistence + auto-retrain trigger |
| `rag_evaluator.py` | RAGAS evaluation pipeline |
| `synthetic_conversation_generator.py` | OOD synthetic data generator |
| `retrain_from_conversations.py` | Retrain intent model from logged conversations |
| `intents_enhanced_2.csv` | Training dataset (14 intent classes) |
| `templates/` | Jinja2 HTML templates (chat UI, agent dashboard, admin panel) |

## Configuration

All configuration is via environment variables in `.env`. See `.env.example` for the full reference with descriptions.

Key settings:

```env
# LLMs
MISTRAL_API_KEY=...
GEMINI_API_KEY=...

# Vector DB (leave blank to use local disk store)
QDRANT_URL=...
QDRANT_API_KEY=...

# Tuning
TEMPLATE_CONFIDENCE=0.75      # confidence threshold for zero-latency template responses
HYBRID_KEYWORD_WEIGHT=0.5     # TF-IDF weight in hybrid retrieval
HYBRID_SEMANTIC_WEIGHT=0.5    # Qdrant semantic weight in hybrid retrieval
AUTO_RETRAIN_THRESHOLD=10     # retrain after this many new conversation pairs
```

## Retraining the Intent Model

```bash
python intent_model.py
```

This trains from `intents_enhanced_2.csv`, prints a CV accuracy report, and saves a new `.pkl` file. Update `MODEL_PATH` in `.env` to use the new model.

## Evaluation

Visit `/admin` in your browser, or trigger via API:

```bash
curl -X POST http://localhost:8000/admin/evaluate
curl http://localhost:8000/admin/scores
```

## License

MIT
