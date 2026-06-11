"""
retrain_from_conversations.py
─────────────────────────────
Reads live_conversations.csv + intents_enhanced_2.csv, labels new
utterances (via Gemini or rule-based fallback), appends them to the
training CSV, expands RAG policies from agent responses, and retrains
the intent model — hot-swapping it in the running app without restart.

Usage
-----
  # standalone
  python retrain_from_conversations.py

  # from app via /admin/retrain endpoint (passes app globals in)
  from retrain_from_conversations import run_retrain
  run_retrain(classifier=classifier, rag_engine=rag_engine)
"""

import os
import logging
import time
import re
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np

from conversation_logger import load_conversations, LIVE_CONV_CSV
from intent_model import IntentClassifier

logger = logging.getLogger(__name__)

TRAINING_CSV   = os.getenv('TRAINING_CSV',   'intents_enhanced_2.csv')
MODEL_DIR      = os.getenv('MODEL_DIR',      '.')
MIN_NEW_ROWS   = int(os.getenv('MIN_NEW_ROWS', '5'))   # skip retrain if fewer new rows

# ── Known intents ──────────────────────────────────────────────────────────────
KNOWN_INTENTS = [
    'order_status', 'return_refund', 'cancellation', 'billing_issue',
    'technical_support', 'account_help', 'appointment_booking', 'product_inquiry',
    'feedback', 'greeting', 'goodbye', 'thanks', 'shipping_info',
    'human_agent', 'out_of_scope',
]

# ── Keyword-based intent labeller (fast, no API call needed) ──────────────────
INTENT_KEYWORDS = {
    'order_status':       ['order status','track','tracking','where is my order','package','shipment','delivery','dispatched','in transit','arrived','not delivered'],
    'return_refund':      ['return','refund','money back','exchange','send back','damaged','wrong item','defective','store credit'],
    'cancellation':       ['cancel','cancellation','stop order','do not ship','undo order','recall'],
    'billing_issue':      ['billing','charge','payment','invoice','overcharged','duplicate charge','card declined','payment failed','wrong charge'],
    'technical_support':  ['not working','error','crash','bug','slow','broken','cant login','cant sign in','website down','app crash','white screen','blank page'],
    'account_help':       ['password','reset','account','login','2fa','two factor','profile','username','email','locked out','forgot password'],
    'appointment_booking':['appointment','book','schedule','reserve','slot','consultation','reschedule','cancel appointment'],
    'product_inquiry':    ['product','item','specs','features','color','size','in stock','warranty','compare','recommend','ingredients'],
    'feedback':           ['feedback','review','complaint','compliment','suggestion','rate','experience','survey','report a problem'],
    'greeting':           ['hello','hi','hey','good morning','good afternoon','good evening','help','start','begin'],
    'goodbye':            ['bye','goodbye','see you','farewell','all done','nothing else','take care','signing off','end chat'],
    'thanks':             ['thank','thanks','appreciate','grateful','cheers','well done','great help','helpful'],
    'shipping_info':      ['shipping','delivery cost','free shipping','express','overnight','international shipping','customs','lost package','shipping address'],
    'human_agent':        ['human','agent','real person','speak to','talk to','manager','supervisor','escalate','live agent','call center','phone number'],
    'out_of_scope':       ['weather','joke','news','stock','bitcoin','sports','recipe','medical','legal','translate','poem','homework','calculate','movie'],
}


def label_utterance_rule(text: str) -> str:
    """Fast keyword-based intent labelling — no API needed."""
    text_lower = text.lower()
    scores = defaultdict(int)
    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[intent] += 1
    if not scores:
        return 'out_of_scope'
    return max(scores, key=scores.get)


def label_utterance_llm(text: str, agent_response: str, llm_model) -> str:
    """
    Use Gemini to label the intent — falls back to rule-based if API fails.
    """
    try:
        prompt = f"""You are an intent classification expert for a customer support chatbot.
Given a customer message and the agent's response, classify the customer message into
EXACTLY ONE of these intents:

{', '.join(KNOWN_INTENTS)}

Customer message: "{text}"
Agent response: "{agent_response}"

Reply with ONLY the intent label, nothing else."""
        resp = llm_model.generate_content(prompt)
        label = resp.text.strip().lower().replace(' ', '_')
        # Validate it's a known intent
        if label in KNOWN_INTENTS:
            return label
        # Try partial match
        for intent in KNOWN_INTENTS:
            if intent in label:
                return intent
    except Exception as e:
        logger.warning(f"LLM labelling failed: {e} — using rule-based fallback")
    return label_utterance_rule(text)


def extract_policy_snippets(agent_responses: list, intent: str) -> list:
    """
    Extract unique, high-quality agent responses to use as policy examples.
    Filters out very short or generic responses.
    """
    MIN_LEN = 30
    seen = set()
    snippets = []
    for resp in agent_responses:
        resp = resp.strip()
        if len(resp) < MIN_LEN:
            continue
        # Deduplicate by normalised form
        norm = re.sub(r'\s+', ' ', resp.lower())
        if norm in seen:
            continue
        seen.add(norm)
        snippets.append(resp)
    return snippets[:10]  # cap at 10 per intent


def build_new_training_rows(conversations: list, llm_model=None) -> list:
    """
    Convert raw conversation pairs into training rows compatible with
    intents_enhanced_2.csv schema.
    Returns list of dicts ready to append to the CSV.
    """
    rows = []
    for i, conv in enumerate(conversations):
        utterance = conv.get('utterance', '').strip()
        response  = conv.get('response',  '').strip()
        if not utterance:
            continue

        # Label intent
        if llm_model and response:
            intent = label_utterance_llm(utterance, response, llm_model)
        else:
            intent = label_utterance_rule(utterance)

        rows.append({
            'conversation_id':    f'live_{conv.get("session_id","x")}_{conv.get("turn",i)}',
            'user_id':            f'live_user_{conv.get("session_id","x")}',
            'session_id':         conv.get('session_id', f'live_sess_{i}'),
            'timestamp':          conv.get('timestamp', datetime.now().isoformat()),
            'channel':            conv.get('channel', 'live_chat'),
            'language':           'en',
            'device':             'unknown',
            'user_type':          'returning',
            'conversation_turn':  conv.get('turn', 1),
            'previous_utterance': '',
            'utterance':          utterance,
            'extracted_entities': '{}',
            'intent':             intent,
            'topic_category':     'live_agent',
            'sentiment':          'neutral',
            'urgency':            'medium',
            'bot_response':       response,
            'previous_bot_response': '',
            'suggested_action':   'live_agent_response',
            'is_escalated':       'yes',
            'escalation_priority':'P2',
            'confidence_score':   0.95,
            'response_time_ms':   int(conv.get('duration_s', 1000)),
        })
    return rows


def append_to_training_csv(new_rows: list, csv_path: str = TRAINING_CSV) -> int:
    """
    Append new labelled rows to the training CSV.
    Skips rows whose utterance already exists in the file.
    Returns number of rows actually appended.
    """
    if not new_rows:
        return 0

    # Load existing utterances to avoid duplicates
    existing = set()
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path, encoding='latin-1', usecols=['utterance'])
        existing = set(df_existing['utterance'].str.lower().str.strip().tolist())

    # Filter duplicates
    to_append = [r for r in new_rows if r['utterance'].lower().strip() not in existing]
    if not to_append:
        logger.info("No new unique utterances to append.")
        return 0

    df_new = pd.DataFrame(to_append)

    # Get fieldnames from existing CSV header
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='latin-1') as f:
            fieldnames = f.readline().strip().split(',')
        # Align columns
        for col in fieldnames:
            if col not in df_new.columns:
                df_new[col] = ''
        df_new = df_new[fieldnames]
        df_new.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8')
    else:
        df_new.to_csv(csv_path, index=False, encoding='utf-8')

    logger.info(f"Appended {len(to_append)} new rows to {csv_path}")
    return len(to_append)


def expand_rag_policies(conversations: list, rag_engine, llm_model=None):
    """
    Group agent responses by intent and inject high-quality examples
    into the RAG knowledge base as additional policy context.
    This enriches the bot's responses without requiring a full retrain.
    """
    # Group agent responses by intent
    by_intent = defaultdict(list)
    for conv in conversations:
        utterance = conv.get('utterance', '').strip()
        response  = conv.get('response',  '').strip()
        if not utterance or not response:
            continue
        if llm_model:
            intent = label_utterance_llm(utterance, response, llm_model)
        else:
            intent = label_utterance_rule(utterance)
        by_intent[intent].append(response)

    updated = []
    for intent, responses in by_intent.items():
        snippets = extract_policy_snippets(responses, intent)
        if not snippets:
            continue

        policy_obj = rag_engine.knowledge_base.policies.get(intent)
        if policy_obj:
            # Append real agent examples to the existing policy
            examples_block = "\n\nREAL AGENT RESPONSES (learned from live chats):\n" + \
                             "\n".join(f"• {s}" for s in snippets[:5])
            # Only add if not already there
            if "REAL AGENT RESPONSES" not in policy_obj['policy']:
                policy_obj['policy'] += examples_block
            else:
                # Replace the block with fresh examples
                base = policy_obj['policy'].split("\n\nREAL AGENT RESPONSES")[0]
                policy_obj['policy'] = base + examples_block

            # Add new keywords from agent responses
            new_kws = _extract_keywords(snippets)
            existing_kws = set(policy_obj['keywords'])
            policy_obj['keywords'] = list(existing_kws | new_kws)
            updated.append(intent)

    if updated:
        # Rebuild the TF-IDF index with enriched policies
        rag_engine.knowledge_base._build_indices()
        logger.info(f"RAG policies expanded for intents: {updated}")

    return updated


def _extract_keywords(texts: list) -> set:
    """Pull meaningful words from agent responses as new keywords."""
    stopwords = {
        'i', 'me', 'my', 'we', 'our', 'you', 'your', 'the', 'a', 'an', 'is',
        'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'up', 'about',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
        'not', 'no', 'can', 'it', 'its', 'this', 'that', 'these', 'those',
        'if', 'as', 'than', 'then', 'when', 'where', 'which', 'who', 'whom',
        'what', 'how', 'all', 'any', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'only', 'own', 'same', 'too', 'very', 'just', 'also',
        'let', 'get', 'got', 'go', 'going', 'please', 'thank', 'thanks',
    }
    words = set()
    for text in texts:
        for word in re.findall(r'\b[a-z]{4,}\b', text.lower()):
            if word not in stopwords:
                words.add(word)
    return words


def retrain_model(csv_path: str = TRAINING_CSV, classifier: IntentClassifier = None) -> str:
    """
    Retrain the intent classifier on the updated CSV.
    Returns the path to the new model file.
    """
    if classifier is None:
        classifier = IntentClassifier()

    timestamp = int(time.time())
    model_file = os.path.join(MODEL_DIR, f'intent_model_v2_{timestamp}.pkl')

    logger.info(f"Retraining on {csv_path} …")
    X, y = classifier.load_data(csv_path)
    classifier.train(X, y)
    classifier.save_model(model_file)
    logger.info(f"New model saved: {model_file}")
    return model_file


def hot_swap_model(new_model_path: str, app_classifier: IntentClassifier):
    """
    Load the new model into the running classifier in-place so the app
    picks it up without a restart.
    """
    app_classifier.load_model(new_model_path)
    # Update the .env MODEL_PATH so future restarts use the new model
    _update_env('MODEL_PATH', os.path.basename(new_model_path))
    logger.info(f"Hot-swapped model to {new_model_path}")


def _update_env(key: str, value: str, env_file: str = '.env'):
    """Update a key=value line in the .env file."""
    if not os.path.exists(env_file):
        return
    with open(env_file, 'r') as f:
        lines = f.readlines()
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f'{key}='):
            lines[i] = f'{key}={value}\n'
            updated = True
            break
    if not updated:
        lines.append(f'{key}={value}\n')
    with open(env_file, 'w') as f:
        f.writelines(lines)


# ── Main entry point ───────────────────────────────────────────────────────────

def run_retrain(classifier=None, rag_engine=None, llm_model=None, force: bool = False):
    """
    Full pipeline:
      1. Load live conversations
      2. Label intents
      3. Append to training CSV
      4. Expand RAG policies
      5. Retrain model
      6. Hot-swap into running app

    If classifier and rag_engine are None, tries to import from app globals.
    
    Returns a summary dict.
    """
    summary = {
        'status': 'ok',
        'conversations_loaded': 0,
        'new_rows_appended': 0,
        'policies_expanded': [],
        'new_model': None,
        'skipped': False,
        'error': None,
    }

    try:
        # Try to get app globals if not provided
        if classifier is None or rag_engine is None:
            try:
                import app as app_module
                if classifier is None:
                    classifier = app_module.classifier
                if rag_engine is None:
                    rag_engine = app_module.rag_engine
                if llm_model is None and hasattr(app_module, 'gemini_model'):
                    llm_model = app_module.gemini_model
            except Exception as e:
                logger.warning(f"Could not import app globals: {e}")

        # 1. Load
        conversations = load_conversations()
        summary['conversations_loaded'] = len(conversations)
        logger.info(f"Loaded {len(conversations)} conversation pairs from {LIVE_CONV_CSV}")

        if not conversations and not force:
            summary['skipped'] = True
            summary['status'] = 'skipped'
            summary['reason'] = 'No live conversations logged yet'
            return summary

        # 2 & 3. Label + append to CSV
        new_rows = build_new_training_rows(conversations, llm_model=llm_model)
        appended = append_to_training_csv(new_rows)
        summary['new_rows_appended'] = appended

        if appended < MIN_NEW_ROWS and not force:
            summary['skipped'] = True
            summary['status'] = 'skipped'
            summary['reason'] = f'Only {appended} new rows (minimum {MIN_NEW_ROWS})'
            logger.info(summary['reason'])
        else:
            # 5. Retrain
            new_model_path = retrain_model(csv_path=TRAINING_CSV, classifier=classifier)
            summary['new_model'] = new_model_path

            # 6. Hot-swap
            if classifier is not None:
                hot_swap_model(new_model_path, classifier)

        # 4. Expand RAG policies (always, even if we skip retrain)
        if rag_engine is not None and conversations:
            expanded = expand_rag_policies(conversations, rag_engine, llm_model=llm_model)
            summary['policies_expanded'] = expanded

    except Exception as e:
        logger.error(f"Retrain pipeline error: {e}", exc_info=True)
        summary['status'] = 'error'
        summary['error'] = str(e)

    return summary


# ── Standalone execution ───────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    from dotenv import load_dotenv
    load_dotenv()
    llm = None
    api_key = os.getenv('GEMINI_API_KEY', '')
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            llm = genai.GenerativeModel(
                model_name=os.getenv('GEMINI_MODEL', 'gemini-2.5-flash'),
                generation_config={'temperature': 0.1, 'max_output_tokens': 50},
            )
            print("Gemini LLM enabled for intent labelling")
        except Exception as e:
            print(f"Gemini unavailable ({e}) — using rule-based labelling")

    result = run_retrain(llm_model=llm, force=True)
    print("\n── Retrain Summary ──────────────────────────────")
    for k, v in result.items():
        print(f"  {k}: {v}")
