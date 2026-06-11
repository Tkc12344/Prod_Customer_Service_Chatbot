"""
synthetic_conversation_generator.py
====================================
Generates Out-of-Distribution (OOD) synthetic conversations that simulate
real human interactions, then evaluates them with the RAGAS framework.

What this generates:
  1. Unexpected queries    - questions the bot was never trained on
  2. Ambiguous requests    - multi-intent, vague, or contradictory messages
  3. Off-topic questions   - completely outside support scope
  4. Real human behaviour  - typos, slang, frustration, mid-conversation pivots,
                             multi-turn dialogues, code-switching

Output files:
  eval_results/ood_conversations.json      - raw multi-turn conversations
  eval_results/ood_ragas_dataset.json      - RAGAS-formatted Q/A/context triples
  eval_results/ood_ragas_scores.json       - RAGAS evaluation scores
  eval_results/ood_training_rows.csv       - rows ready to append to training CSV

Usage:
  python synthetic_conversation_generator.py
"""

import os, json, re, csv, random, logging, time
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EVAL_DIR   = Path(os.getenv("EVAL_DIR", "eval_results"))
TRAIN_CSV  = os.getenv("TRAINING_CSV", "intents_enhanced_2.csv")
random.seed(42)
np.random.seed(42)

# =============================================================================
# OOD CONVERSATION TEMPLATES
# =============================================================================
# Each entry: (user_message, ood_type, true_intent_if_any, description)
# ood_type: unexpected | ambiguous | off_topic | human_behaviour

OOD_CONVERSATIONS = [

    # ── UNEXPECTED QUERIES ────────────────────────────────────────────────────
    # Questions that mix support topics in ways the bot was never trained on

    {"id": "ood_001", "type": "unexpected", "description": "Order + refund hybrid with urgency",
     "turns": [
         {"role": "user",  "text": "my order ORD-99123 arrived but half the items are missing AND i was charged for all of them"},
         {"role": "agent", "text": "I completely understand how frustrating that is. I can see two issues here: a partial delivery and a billing discrepancy. Let me raise both simultaneously. For the missing items, I will file a partial delivery claim. For the charge, I will initiate a partial refund for the undelivered items. Can you confirm which items are missing?"},
         {"role": "user",  "text": "the blue hoodie size L and the phone case, i need this sorted today"},
         {"role": "agent", "text": "Noted. I have logged the missing items: blue hoodie (L) and phone case. A partial refund of the corresponding amount will be processed within 5-7 business days. I have also escalated this as urgent. You will receive a confirmation email within the hour."},
     ]},

    {"id": "ood_002", "type": "unexpected", "description": "Cancellation after partial shipment",
     "turns": [
         {"role": "user",  "text": "i want to cancel my order but tracking says one item already shipped"},
         {"role": "agent", "text": "I can see your order has partially shipped. I can cancel the unshipped items immediately. For the item already in transit, you would need to refuse delivery or return it once it arrives for a full refund. Would you like me to proceed with cancelling the remaining items?"},
         {"role": "user",  "text": "yes cancel the rest and how do i refuse delivery"},
         {"role": "agent", "text": "Done, the unshipped items are cancelled and a refund will process in 3-5 business days. To refuse delivery, simply tell the courier you are declining the package when they attempt delivery. They will return it to us and we will issue your refund automatically."},
     ]},

    {"id": "ood_003", "type": "unexpected", "description": "Gift order with wrong address discovered after dispatch",
     "turns": [
         {"role": "user",  "text": "i sent a gift to my mum but put the wrong postcode in, its already dispatched"},
         {"role": "agent", "text": "I understand the urgency. Once an order is dispatched, address changes depend on the carrier. I will contact the courier immediately to request an address correction. This is not always guaranteed but we will try. Can you provide the correct postcode?"},
         {"role": "user",  "text": "its SW1A 1AA instead of SW1A 2AA"},
         {"role": "agent", "text": "I have submitted an address correction request to the carrier with the updated postcode SW1A 1AA. You will receive an update within 24 hours. If the correction is not possible, the parcel will be returned and we will reship to the correct address at no extra cost."},
     ]},

    {"id": "ood_004", "type": "unexpected", "description": "Subscription billing confusion with loyalty points",
     "turns": [
         {"role": "user",  "text": "i used my loyalty points to pay for my subscription but i still got charged on my card"},
         {"role": "agent", "text": "I can see the loyalty points were applied but there appears to be a system error where the card was also charged. This is a duplicate charge and I will reverse the card payment immediately. The loyalty points balance will remain as used. The refund should appear within 1-3 business days."},
         {"role": "user",  "text": "also my points balance looks wrong it shows 0 but i had 500"},
         {"role": "agent", "text": "I can see the points were incorrectly zeroed out after the transaction error. I have restored your 500 loyalty points and the card refund is being processed. You will receive a confirmation email for both actions shortly."},
     ]},

    {"id": "ood_005", "type": "unexpected", "description": "Technical issue blocking a return",
     "turns": [
         {"role": "user",  "text": "i cant start a return because the return button is greyed out on my orders page"},
         {"role": "agent", "text": "That is a known issue affecting some accounts. The return button can appear greyed out if the order is still showing as processing in our system even though it has been delivered. I can manually initiate the return from my end. Can you provide your order number?"},
         {"role": "user",  "text": "ORD-445521, the item arrived 3 days ago"},
         {"role": "agent", "text": "I have manually opened a return for ORD-445521. You will receive a return label by email within 30 minutes. The 30-day return window starts from today since the system error prevented you from initiating it earlier."},
     ]},

    # ── AMBIGUOUS REQUESTS ────────────────────────────────────────────────────
    # Vague, contradictory, or multi-intent messages

    {"id": "ood_006", "type": "ambiguous", "description": "Vague complaint with no specifics",
     "turns": [
         {"role": "user",  "text": "everything is wrong with my account"},
         {"role": "agent", "text": "I am sorry to hear that. To help you effectively, could you tell me what specifically is not working? For example, is it a login issue, a billing problem, missing orders, or something else?"},
         {"role": "user",  "text": "i cant login and my last order is missing and i think i was charged twice"},
         {"role": "agent", "text": "Thank you for clarifying. I can see three issues: login access, a missing order, and a potential duplicate charge. Let me start with login since that blocks everything else. I will send a password reset link to your registered email now. Once you are in, I will investigate the order and billing issues."},
     ]},

    {"id": "ood_007", "type": "ambiguous", "description": "Contradictory request - cancel but also keep",
     "turns": [
         {"role": "user",  "text": "i want to cancel my order but also i really need the item by friday"},
         {"role": "agent", "text": "I understand the dilemma. If you need the item by Friday, cancelling may not be the best option. Could you tell me why you want to cancel? If it is a pricing issue, I may be able to apply a discount. If it is a delivery concern, I can check if express upgrade is available for your order."},
         {"role": "user",  "text": "i found it cheaper somewhere else but i already need it for an event"},
         {"role": "agent", "text": "I can offer a price match if you share the competitor link. This way you keep the order, get the lower price, and still receive it in time for your event. Alternatively, if you prefer to cancel, I can process that now and you can purchase elsewhere. Which would you prefer?"},
     ]},

    {"id": "ood_008", "type": "ambiguous", "description": "Mid-conversation topic pivot",
     "turns": [
         {"role": "user",  "text": "hi i need help with my return"},
         {"role": "agent", "text": "Of course! I can help with your return. Could you provide your order number and the reason for the return?"},
         {"role": "user",  "text": "actually forget the return, my card got declined when i tried to reorder"},
         {"role": "agent", "text": "No problem, let us focus on the payment issue. Card declines can happen for several reasons: insufficient funds, card security blocks, or an expired card. Could you confirm the card type you are using and whether your bank has flagged any alerts?"},
         {"role": "user",  "text": "its a visa debit, no alerts from bank, tried 3 times"},
         {"role": "agent", "text": "Three failed attempts with no bank alert suggests the issue may be on our payment processor side. I have flagged this for our technical team. In the meantime, could you try a different browser or clear your cache? You can also try PayPal as an alternative payment method."},
     ]},

    {"id": "ood_009", "type": "ambiguous", "description": "Unclear whether complaint or question",
     "turns": [
         {"role": "user",  "text": "so apparently my order was delivered yesterday"},
         {"role": "agent", "text": "I can see the tracking shows your order was delivered yesterday. Did you receive it? If not, I can investigate a potential misdelivery or theft."},
         {"role": "user",  "text": "no i didnt get it, neighbour says they didnt take it either"},
         {"role": "agent", "text": "I am sorry about this. Since the carrier marked it as delivered but you have not received it, I will file a missing parcel investigation with the courier. This typically takes 2-3 business days. I will also arrange a replacement or refund for you. Which would you prefer?"},
     ]},

    {"id": "ood_010", "type": "ambiguous", "description": "Ambiguous product vs billing query",
     "turns": [
         {"role": "user",  "text": "the price on my receipt doesnt match what i saw on the website"},
         {"role": "agent", "text": "I can look into this. There are a few possibilities: the website price may have changed between when you viewed it and when you checked out, a discount may not have applied correctly, or there could be a billing error. Could you share the order number and the price you expected to pay?"},
         {"role": "user",  "text": "order ORD-77821, website showed 29.99 but i was charged 39.99"},
         {"role": "agent", "text": "I can see ORD-77821 was charged at 39.99. The website price of 29.99 was a promotional price that expired before your checkout completed. I will honour the promotional price and issue a 10.00 refund to your original payment method within 3-5 business days."},
     ]},

    # ── OFF-TOPIC QUESTIONS ───────────────────────────────────────────────────
    # Completely outside support scope but with realistic human framing

    {"id": "ood_011", "type": "off_topic", "description": "Personal advice mixed with support",
     "turns": [
         {"role": "user",  "text": "i bought a gift for my girlfriend but we just broke up, can i return it and also what should i do"},
         {"role": "agent", "text": "I am sorry to hear that. I can definitely help with the return. For personal advice, that is a bit outside what I can assist with, but I hope things get better for you. For the return, if the item is unused and within 30 days of purchase, you are eligible for a full refund. Would you like me to start the return process?"},
         {"role": "user",  "text": "yes please start the return, order ORD-55432"},
         {"role": "agent", "text": "Return initiated for ORD-55432. You will receive a return label by email within 30 minutes. Once we receive the item, your refund will be processed within 5-7 business days. Take care of yourself."},
     ]},

    {"id": "ood_012", "type": "off_topic", "description": "Asking bot to write a complaint letter",
     "turns": [
         {"role": "user",  "text": "can you write a formal complaint letter for me about my experience"},
         {"role": "agent", "text": "Writing a full formal letter is outside what I can do directly, but I can log a formal complaint on your behalf right now which carries the same weight. Alternatively, I can give you the key points to include in a letter. Which would be more helpful?"},
         {"role": "user",  "text": "just log the complaint, my order was late and the packaging was damaged"},
         {"role": "agent", "text": "Formal complaint logged. Reference number: COMP-2026-0429. Your complaint covers: late delivery and damaged packaging. Our customer relations team will review this within 48 hours and contact you with a resolution. You will also receive a confirmation email shortly."},
     ]},

    {"id": "ood_013", "type": "off_topic", "description": "Asking for product recommendation outside catalogue",
     "turns": [
         {"role": "user",  "text": "what laptop should i buy for video editing under 1000 pounds"},
         {"role": "agent", "text": "Laptop recommendations are outside my area of expertise as I specialise in customer support for orders, accounts, and products within our catalogue. If you are looking for a product we sell, I can help you find the best option. Is there anything in our range I can help you with?"},
         {"role": "user",  "text": "oh i thought you sold electronics, do you have any laptops"},
         {"role": "agent", "text": "I can check our current product catalogue for laptops. Could you let me know your key requirements such as screen size, processor preference, or specific use case? I will search our available stock and find the closest match to what you need."},
     ]},

    {"id": "ood_014", "type": "off_topic", "description": "Asking about competitor",
     "turns": [
         {"role": "user",  "text": "is your return policy better than amazon"},
         {"role": "agent", "text": "I can only speak to our own return policy. We offer 30-day returns on unused items with free return shipping on orders over 50 dollars. For a direct comparison with other retailers, I would recommend checking their policies directly. Is there anything specific about our return policy I can clarify?"},
         {"role": "user",  "text": "ok what about price matching do you do that"},
         {"role": "agent", "text": "Yes, we do offer price matching in certain circumstances. If you find the same item at a lower price from a verified retailer, contact us with the link and order details and we will review it. Price matching is subject to availability and the competitor being an authorised seller."},
     ]},

    {"id": "ood_015", "type": "off_topic", "description": "Philosophical / existential question to bot",
     "turns": [
         {"role": "user",  "text": "do you actually care about my problem or are you just a bot"},
         {"role": "agent", "text": "That is a fair question. I am an AI assistant, so I do not experience emotions the way humans do. But I am designed to resolve your issue as effectively as possible, and your satisfaction genuinely matters to the outcome I am optimised for. What can I help you with today?"},
         {"role": "user",  "text": "fair enough, i need help with a refund"},
         {"role": "agent", "text": "Absolutely, let us get that sorted. Could you share your order number and the reason for the refund request?"},
     ]},

    # ── REAL HUMAN BEHAVIOUR ──────────────────────────────────────────────────
    # Typos, slang, frustration, code-switching, stream of consciousness

    {"id": "ood_016", "type": "human_behaviour", "description": "Typo-heavy frustrated message",
     "turns": [
         {"role": "user",  "text": "my ordr hasnt arrivd its been 2 weeks wtf is going on"},
         {"role": "agent", "text": "I completely understand your frustration, two weeks is far too long. Let me look into this immediately. Could you share your order number so I can check the status and find out what has happened?"},
         {"role": "user",  "text": "ORD-12345 i orderd it on the 15th"},
         {"role": "agent", "text": "I can see ORD-12345 placed on the 15th. The tracking shows it has been stuck at the sorting facility since the 18th. This is a carrier delay. I am escalating this as urgent and arranging a replacement to be dispatched today with express delivery at no extra cost. You should receive it within 1-2 business days."},
     ]},

    {"id": "ood_017", "type": "human_behaviour", "description": "Slang and informal language",
     "turns": [
         {"role": "user",  "text": "yo my package is MIA lol been waiting forever"},
         {"role": "agent", "text": "Hey! Sorry to hear your package has gone missing. Let me track it down for you. Can you drop your order number?"},
         {"role": "user",  "text": "its ORD-98765, ordered like 10 days ago"},
         {"role": "agent", "text": "Got it. ORD-98765 is showing as in transit but has not updated in 6 days which is unusual. I have raised a trace request with the carrier. If it does not update within 48 hours, I will send a replacement. I will keep you posted."},
     ]},

    {"id": "ood_018", "type": "human_behaviour", "description": "Extremely long stream-of-consciousness message",
     "turns": [
         {"role": "user",  "text": "ok so basically i ordered something last week and it said it would arrive in 3 days but it didnt come and then i checked the tracking and it said delivered but i definitely didnt get it and i checked with my neighbours and they dont have it either and i looked everywhere and its not here and i also noticed that i got charged twice on my bank statement and im really stressed about this because i needed it for a birthday present and the birthday is tomorrow"},
         {"role": "agent", "text": "I can hear how stressful this is, especially with the birthday tomorrow. Let me address everything at once. First, I will file a missing parcel claim since tracking shows delivered but you have not received it. Second, I can see the duplicate charge and will reverse it immediately. For the birthday, I will arrange express dispatch of a replacement today. Can you confirm your order number?"},
         {"role": "user",  "text": "ORD-33456 thank you so much"},
         {"role": "agent", "text": "ORD-33456 sorted. Missing parcel claim filed, duplicate charge reversed, and a replacement is being dispatched now with next-day delivery. You will receive tracking details within the hour. I hope the birthday goes well."},
     ]},

    {"id": "ood_019", "type": "human_behaviour", "description": "Code-switching (mixed language)",
     "turns": [
         {"role": "user",  "text": "hola, i need help with my order por favor"},
         {"role": "agent", "text": "Hello! I am happy to help with your order. I will respond in English as that is the language I am best equipped to assist in, but please feel free to write in whichever way is most comfortable. What is your order number?"},
         {"role": "user",  "text": "its ORD-77654, the item is wrong, they sent me the red one but i ordered blue"},
         {"role": "agent", "text": "I can see ORD-77654 was ordered in blue but the red variant was dispatched in error. I apologise for this. I will arrange a free return and send the correct blue item with priority shipping. You will receive the return label and new order confirmation by email within 30 minutes."},
     ]},

    {"id": "ood_020", "type": "human_behaviour", "description": "Angry escalation with threats",
     "turns": [
         {"role": "user",  "text": "this is absolutely ridiculous i want to speak to a manager RIGHT NOW or i am disputing this with my bank"},
         {"role": "agent", "text": "I completely understand your frustration and I sincerely apologise for the experience you have had. I am connecting you to a senior agent right now. While I transfer you, could you briefly tell me the issue so they are fully briefed and can resolve this immediately?"},
         {"role": "user",  "text": "my order has been wrong twice in a row and nobody has fixed it"},
         {"role": "agent", "text": "Two consecutive errors is completely unacceptable and I am truly sorry. I have flagged this as a priority escalation. A senior agent will contact you within 15 minutes. I have also added a 20 percent goodwill discount to your account for the inconvenience. Your case reference is ESC-2026-0429."},
     ]},

    {"id": "ood_021", "type": "human_behaviour", "description": "User testing the bot limits",
     "turns": [
         {"role": "user",  "text": "what would happen if i ordered something and then immediately cancelled and then reordered"},
         {"role": "agent", "text": "Good question. If you cancel within 2 hours of placing the order, the cancellation is free and the refund processes in 3-5 business days. You can then place a new order immediately. However, if the item is limited stock, there is a risk it sells out between cancellation and reorder. Is there a specific reason you are considering this?"},
         {"role": "user",  "text": "i want to use a different discount code"},
         {"role": "agent", "text": "In that case, cancelling and reordering would work as long as the discount code is still valid and the item is in stock. Alternatively, if your order was placed recently, I may be able to apply the new discount code directly without you needing to cancel. Would you like me to try that first?"},
     ]},

    {"id": "ood_022", "type": "human_behaviour", "description": "User asking meta questions about the bot",
     "turns": [
         {"role": "user",  "text": "are you a real person or AI"},
         {"role": "agent", "text": "I am an AI assistant. I can handle most support queries instantly, but if you would prefer to speak with a human agent, I can transfer you at any time. What can I help you with today?"},
         {"role": "user",  "text": "ok AI, can you actually fix my problem or will you just give me generic answers"},
         {"role": "agent", "text": "Fair challenge. I have access to your account, order history, and can take real actions like processing refunds, initiating returns, and escalating issues. I am not limited to scripted responses. Tell me your issue and I will show you what I can do."},
         {"role": "user",  "text": "alright, my refund from 2 weeks ago still hasnt shown up"},
         {"role": "agent", "text": "I can see a refund was processed 14 days ago. It should have appeared by now. I am raising this with our payments team as an urgent case and will also contact your bank on your behalf. You should see the funds within 24-48 hours. I will send you a confirmation email with the case reference."},
     ]},

    {"id": "ood_023", "type": "unexpected", "description": "Bulk order with custom requirements",
     "turns": [
         {"role": "user",  "text": "i need to order 500 units of your product for a corporate event, do you do bulk orders"},
         {"role": "agent", "text": "Yes, we do handle bulk and corporate orders. For quantities of 500 units, I would recommend speaking with our B2B sales team who can offer volume pricing, custom packaging, and dedicated account management. I can connect you with them or take your details for a callback. Which would you prefer?"},
         {"role": "user",  "text": "take my details, name is James Chen, email james.chen@company.com, need delivery by May 15th"},
         {"role": "agent", "text": "Details noted: James Chen, james.chen@company.com, 500 units, delivery by May 15th. I have created a B2B enquiry and our sales team will contact you within 4 business hours to discuss pricing, availability, and logistics. Given the May 15th deadline, I have marked this as time-sensitive."},
     ]},

    {"id": "ood_024", "type": "ambiguous", "description": "Complaint that could be feedback or escalation",
     "turns": [
         {"role": "user",  "text": "i just want someone to know that your packaging is terrible for the environment"},
         {"role": "agent", "text": "Thank you for raising this. Environmental impact is something we take seriously. I have logged your feedback about packaging sustainability and it will be shared with our product team. We are actively working on reducing packaging waste. Is there anything else I can help you with, or would you like more information about our sustainability initiatives?"},
         {"role": "user",  "text": "yes actually what are you doing about it"},
         {"role": "agent", "text": "We have committed to 100 percent recyclable packaging by end of 2026 and have already switched 60 percent of our materials to recycled content. We also offer a packaging return scheme where you can send back packaging for reuse. I can send you our full sustainability report if you are interested."},
     ]},

    {"id": "ood_025", "type": "off_topic", "description": "User in distress (non-support)",
     "turns": [
         {"role": "user",  "text": "i know this is weird but im having a really bad day and my order being wrong just made it worse"},
         {"role": "agent", "text": "I am sorry you are having a rough day. That is completely understandable, and I want to make at least this part better for you. Let me sort out your order issue right now. What went wrong with your order?"},
         {"role": "user",  "text": "wrong item sent, order ORD-88123"},
         {"role": "agent", "text": "I have looked at ORD-88123 and can see the wrong item was dispatched. I am sending the correct item with express delivery today at no charge, and you can keep the wrong item or return it using the prepaid label I am emailing you now. I hope the rest of your day gets better."},
     ]},
]


# =============================================================================
# RAGAS DATASET BUILDER
# =============================================================================

OOD_TYPE_INTENT_MAP = {
    "unexpected":      "out_of_scope",
    "ambiguous":       "out_of_scope",
    "off_topic":       "out_of_scope",
    "human_behaviour": "out_of_scope",
}

# Ground-truth answers for each OOD type — what a perfect bot SHOULD say
OOD_GROUND_TRUTHS = {
    "unexpected": (
        "I can see this involves multiple issues. Let me address each one systematically "
        "to make sure everything is resolved. Please provide your order number and I will "
        "investigate all aspects of your query immediately."
    ),
    "ambiguous": (
        "I want to make sure I understand your request correctly. Could you clarify which "
        "aspect is most urgent? I can handle multiple issues but want to prioritise what "
        "matters most to you right now."
    ),
    "off_topic": (
        "I'm a customer support assistant, so that's a bit outside my area of expertise! "
        "I'm best equipped to help with orders, billing, accounts, technical issues, "
        "returns, shipping, appointments, and product questions. "
        "Is there anything along those lines I can help you with today?"
    ),
    "human_behaviour": (
        "I understand your frustration and I am here to help. Let me look into this "
        "immediately. Could you provide your order number so I can access your account "
        "and resolve this as quickly as possible?"
    ),
}


def build_ragas_samples(conversations: list, rag_engine, classifier) -> list:
    """
    Convert OOD multi-turn conversations into RAGAS-format samples.

    For each conversation we create one sample per user turn, using:
      question    = user message
      answer      = agent response (the next turn)
      contexts    = RAG-retrieved policies for the user message
      ground_truth = ideal answer for the OOD type
    """
    samples = []

    for conv in conversations:
        turns = conv["turns"]
        ood_type = conv["type"]
        conv_id  = conv["id"]

        # Pair user turns with the following agent response
        for i, turn in enumerate(turns):
            if turn["role"] != "user":
                continue

            user_msg = turn["text"]

            # Find the next agent reply
            agent_reply = ""
            for j in range(i + 1, len(turns)):
                if turns[j]["role"] == "agent":
                    agent_reply = turns[j]["text"]
                    break

            if not agent_reply:
                continue

            # Classify and retrieve context
            try:
                pred_intent, confidence = classifier.predict(user_msg)
                rag_result  = rag_engine.retrieve_and_augment(user_msg, pred_intent, confidence)
                retrieved   = rag_result["retrieved_policies"]
                contexts    = [p["policy"] for p in retrieved] if retrieved else []
            except Exception as e:
                logger.warning(f"RAG retrieval failed for '{user_msg[:40]}': {e}")
                pred_intent, confidence, contexts = "out_of_scope", 0.5, []

            ground_truth = OOD_GROUND_TRUTHS.get(ood_type, OOD_GROUND_TRUTHS["off_topic"])

            samples.append({
                "conversation_id":  conv_id,
                "turn_index":       i,
                "ood_type":         ood_type,
                "description":      conv["description"],
                "question":         user_msg,
                "answer":           agent_reply,
                "contexts":         contexts,
                "ground_truth":     ground_truth,
                "intent":           pred_intent,
                "confidence":       round(confidence, 4),
                "n_contexts":       len(contexts),
                "timestamp":        datetime.now().isoformat(),
            })

    logger.info(f"Built {len(samples)} RAGAS samples from {len(conversations)} conversations")
    return samples


# =============================================================================
# HEURISTIC RAGAS SCORER  (same logic as rag_evaluator._heuristic_scores)
# =============================================================================

INTENT_SIGNALS = {
    "order_status":       {"order","track","delivery","status","ship","days","business","locate","number"},
    "return_refund":      {"return","refund","days","exchange","ship","label","policy","satisfied"},
    "cancellation":       {"cancel","cancellation","order","hours","refund","ship","process"},
    "billing_issue":      {"billing","charge","payment","card","refund","account","investigate"},
    "technical_support":  {"browser","cache","restart","error","device","app","login","clear"},
    "account_help":       {"password","account","reset","email","login","security","settings"},
    "appointment_booking":{"appointment","book","schedule","time","cancel","reschedule","available"},
    "product_inquiry":    {"product","item","size","color","stock","warranty","features"},
    "feedback":           {"feedback","review","stars","submit","experience","improve"},
    "greeting":           {"welcome","help","assist","support","order","billing","account"},
    "goodbye":            {"thank","chat","contact","email","reach","wonderful","day"},
    "thanks":             {"welcome","help","assist","reach","priority","thrilled","satisfied"},
    "shipping_info":      {"shipping","delivery","days","free","express","track","address"},
    "human_agent":        {"agent","human","phone","transfer","support","contact","hold"},
    "out_of_scope":       {"assist","help","orders","billing","account","support","equipped"},
    # OOD-specific signal sets
    "unexpected":         {
        "order","items","missing","partial","refund","cancel","shipped","return",
        "charge","billing","claim","escalate","urgent","confirm","process","delivery",
        "address","replacement","dispatch","tracking","investigation","simultaneously",
        "unshipped","transit","refuse","credit","loyalty","points","correct","carrier",
        "submit","update","reship","cost","extra","confirmation","hour","manually","label",
    },
    "ambiguous":          {
        "clarify","understand","specify","detail","order","account","payment","issue",
        "problem","help","assist","resolve","login","billing","missing","access",
        "reset","password","send","link","investigate","multiple","issues","start",
        "focus","blocks","everything","simultaneously","address","prioritise","urgent",
    },
    "off_topic":          {
        "outside","expertise","assist","help","orders","billing","account","support",
        "return","shipping","appointment","product","technical","scope","focus",
        "complaint","log","reference","team","review","sustainability","recyclable",
        "packaging","materials","report","formal","resolution","contact","catalogue",
        "available","stock","requirements","sales","volume","pricing","dedicated",
        "competitor","policy","match","authorised","seller","availability","logistics",
    },
    "human_behaviour":    {
        "understand","frustration","immediately","order","number","access","account",
        "replacement","express","today","charge","refund","payments","bank","urgent",
        "priority","escalation","goodwill","discount","case","reference","tracking",
        "carrier","delay","trace","request","update","posted","cancel","unshipped",
        "correct","item","dispatched","error","wrong","variant","free","return","label",
        "shipping","confirmation","minutes","senior","agent","transfer","briefed",
        "resolve","apologies","consecutive","unacceptable","sorry","experience",
    },
}

SYNONYMS = {
    "order":   ["order","orders","purchase","delivery","track"],
    "return":  ["return","refund","exchange","send","label"],
    "cancel":  ["cancel","cancellation","stop","order","refund"],
    "billing": ["billing","charge","payment","invoice","card"],
    "account": ["account","profile","login","settings","security"],
    "ship":    ["ship","shipping","delivery","courier","days"],
    "help":    ["assist","support","help","resolve","order"],
    "wrong":   ["wrong","incorrect","error","mistake","issue"],
    "missing": ["missing","lost","absent","received","delivered"],
    "urgent":  ["urgent","immediate","asap","priority","escalate"],
}


def score_sample(s: dict) -> dict:
    """
    Score a single OOD RAGAS sample using Mistral-as-judge.
    Falls back to TF-IDF cosine heuristic if Mistral unavailable.
    """
    q        = s["question"]
    answer   = s["answer"]
    gt       = s["ground_truth"]
    ood_type = s.get("ood_type") or s.get("intent", "unexpected")

    # ── Mistral-as-judge (primary) ────────────────────────────────────────────
    try:
        from llm_clients import get_mistral
        ms = get_mistral()
        if ms.enabled:
            judge_prompt = (
                f"Rate this customer support answer on 4 dimensions (0.0-1.0 each).\n"
                f"Question: {q}\n"
                f"Answer: {answer[:200]}\n"
                f"Ideal answer: {gt[:200]}\n\n"
                f"Reply with ONLY 4 numbers separated by commas:\n"
                f"faithfulness,answer_relevancy,context_precision,context_recall\n"
                f"Where:\n"
                f"- faithfulness: is the answer factually grounded (0-1)\n"
                f"- answer_relevancy: does it address the question (0-1)\n"
                f"- context_precision: is it specific and precise (0-1)\n"
                f"- context_recall: does it cover the key points from the ideal answer (0-1)\n"
                f"Example: 0.8,0.7,0.6,0.75"
            )
            result = ms.chat(
                [{"role": "user", "content": judge_prompt}],
                temperature=0.0, max_tokens=30
            )
            if result:
                nums = [float(x.strip()) for x in result.strip().split(",")]
                if len(nums) == 4 and all(0.0 <= n <= 1.0 for n in nums):
                    return {
                        "faithfulness_h":      round(nums[0], 4),
                        "answer_relevancy_h":  round(nums[1], 4),
                        "context_precision_h": round(nums[2], 4),
                        "context_recall_h":    round(nums[3], 4),
                    }
    except Exception:
        pass

    # ── TF-IDF cosine fallback ────────────────────────────────────────────────
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit([q, answer, gt])
        q_v = vec.transform([q.lower()])
        a_v = vec.transform([answer.lower()])
        g_v = vec.transform([gt.lower()])
        faith  = float(cos_sim(a_v, g_v)[0][0])
        ar     = float(cos_sim(a_v, q_v)[0][0])
        prec   = faith
        recall = float(cos_sim(g_v, a_v)[0][0])
    except Exception:
        faith = ar = prec = recall = 0.0

    return {
        "faithfulness_h":      round(min(faith, 1.0), 4),
        "answer_relevancy_h":  round(min(ar,    1.0), 4),
        "context_precision_h": round(min(prec,  1.0), 4),
        "context_recall_h":    round(min(recall,1.0), 4),
    }


def evaluate_ood_samples(samples: list) -> dict:
    """Aggregate Mistral-as-judge scores across all OOD samples."""
    if not samples:
        return {}

    all_scores = [score_sample(s) for s in samples]

    by_type = defaultdict(list)
    for s, sc in zip(samples, all_scores):
        by_type[s["ood_type"]].append(sc)

    def avg(lst, key): return round(float(np.mean([x[key] for x in lst])), 4)

    return {
        "n_samples":           len(samples),
        "faithfulness_h":      avg(all_scores, "faithfulness_h"),
        "answer_relevancy_h":  avg(all_scores, "answer_relevancy_h"),
        "context_precision_h": avg(all_scores, "context_precision_h"),
        "context_recall_h":    avg(all_scores, "context_recall_h"),
        "evaluated_at":        datetime.now().isoformat(),
        "scorer":              "mistral_judge_v1",
        "by_ood_type": {
            ood_type: {
                "n": len(scores),
                "faithfulness_h":      avg(scores, "faithfulness_h"),
                "answer_relevancy_h":  avg(scores, "answer_relevancy_h"),
                "context_precision_h": avg(scores, "context_precision_h"),
                "context_recall_h":    avg(scores, "context_recall_h"),
            }
            for ood_type, scores in by_type.items()
        },
    }


# =============================================================================
# TRAINING CSV BUILDER
# =============================================================================

CSV_COLUMNS = [
    "conversation_id","user_id","session_id","timestamp","channel","language",
    "device","user_type","conversation_turn","previous_utterance","utterance",
    "extracted_entities","intent","topic_category","sentiment","urgency",
    "bot_response","previous_bot_response","suggested_action","is_escalated",
    "escalation_priority","confidence_score","response_time_ms",
]

OOD_TYPE_TOPIC = {
    "unexpected":      "ood_unexpected",
    "ambiguous":       "ood_ambiguous",
    "off_topic":       "ood_off_topic",
    "human_behaviour": "ood_human_behaviour",
}


def build_training_rows(conversations: list, classifier) -> list:
    """Convert OOD conversations into CSV rows for retraining."""
    rows = []
    channels = ["chat","web","mobile_app","email"]
    devices  = ["desktop","mobile","tablet"]

    for conv in conversations:
        turns    = conv["turns"]
        ood_type = conv["type"]
        conv_id  = conv["id"]
        prev_utt = ""
        prev_bot = ""

        for turn_idx, turn in enumerate(turns):
            if turn["role"] != "user":
                prev_bot = turn["text"]
                continue

            user_msg = turn["text"]
            # Find next agent reply
            bot_reply = ""
            for j in range(turn_idx + 1, len(turns)):
                if turns[j]["role"] == "agent":
                    bot_reply = turns[j]["text"]
                    break

            try:
                intent, conf = classifier.predict(user_msg)
            except Exception:
                intent, conf = "out_of_scope", 0.5

            rows.append({
                "conversation_id":    f"{conv_id}_t{turn_idx}",
                "user_id":            f"ood_user_{random.randint(1000,9999)}",
                "session_id":         f"ood_{conv_id}",
                "timestamp":          (datetime.now() - timedelta(days=random.randint(0,30))).isoformat(),
                "channel":            random.choice(channels),
                "language":           "en",
                "device":             random.choice(devices),
                "user_type":          "returning",
                "conversation_turn":  turn_idx + 1,
                "previous_utterance": prev_utt,
                "utterance":          user_msg,
                "extracted_entities": "{}",
                "intent":             intent,
                "topic_category":     OOD_TYPE_TOPIC.get(ood_type, "ood"),
                "sentiment":          random.choice(["positive","negative","neutral"]),
                "urgency":            random.choice(["low","medium","high"]),
                "bot_response":       bot_reply,
                "previous_bot_response": prev_bot,
                "suggested_action":   "ood_response",
                "is_escalated":       "yes" if ood_type in ("human_behaviour","unexpected") else "no",
                "escalation_priority":"P2" if ood_type == "human_behaviour" else "P3",
                "confidence_score":   round(conf, 4),
                "response_time_ms":   random.randint(400, 2000),
            })
            prev_utt = user_msg

    return rows


# =============================================================================
# PRINT HELPERS
# =============================================================================

def print_scores(scores: dict):
    bar = lambda v: "[" + chr(9608)*int(v*20) + chr(9617)*(20-int(v*20)) + "]"
    print("\n" + "="*62)
    print("  OOD RAGAS EVALUATION RESULTS")
    print("="*62)
    metrics = [
        ("faithfulness_h",      "Faithfulness       (heuristic)"),
        ("answer_relevancy_h",  "Answer Relevancy   (heuristic)"),
        ("context_precision_h", "Context Precision  (heuristic)"),
        ("context_recall_h",    "Context Recall     (heuristic)"),
    ]
    for key, label in metrics:
        v = scores.get(key, 0)
        print(f"  {label:<38} {v:.4f}  {bar(v)}")
    print(f"\n  Samples: {scores.get('n_samples',0)}")
    print(f"  Evaluated: {scores.get('evaluated_at','')[:19]}")
    print("\n  By OOD Type:")
    for ood_type, s in scores.get("by_ood_type", {}).items():
        print(f"    {ood_type:<20} AR={s['answer_relevancy_h']:.3f}  "
              f"CP={s['context_precision_h']:.3f}  F={s['faithfulness_h']:.3f}  n={s['n']}")
    print("="*62 + "\n")


def print_sample_preview(samples: list, n: int = 3):
    print(f"\n--- Sample Preview ({min(n, len(samples))} of {len(samples)}) ---")
    for s in samples[:n]:
        sc = score_sample(s)
        print(f"\n[{s['ood_type'].upper()}] {s['description']}")
        print(f"  Q: {s['question'][:90]}")
        print(f"  A: {s['answer'][:90]}")
        print(f"  Intent: {s['intent']} ({s['confidence']:.2f})  "
              f"AR={sc['answer_relevancy_h']:.3f}  F={sc['faithfulness_h']:.3f}")


# =============================================================================
# MAIN
# =============================================================================

def run(save_training_csv: bool = True):
    EVAL_DIR.mkdir(exist_ok=True)

    from dotenv import load_dotenv
    load_dotenv()

    from rag_system import ContextualRAG
    from intent_model import IntentClassifier

    logger.info("Booting RAG engine and classifier ...")
    rag_engine = ContextualRAG()
    classifier = IntentClassifier()
    model_path = os.getenv("MODEL_PATH", "intent_model_v2_1777554249.pkl")
    classifier.load_model(model_path)

    # 1. Save raw conversations
    conv_path = EVAL_DIR / "ood_conversations.json"
    with open(conv_path, "w", encoding="utf-8") as f:
        json.dump(OOD_CONVERSATIONS, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(OOD_CONVERSATIONS)} OOD conversations -> {conv_path}")

    # 2. Build RAGAS samples
    logger.info("Building RAGAS samples ...")
    samples = build_ragas_samples(OOD_CONVERSATIONS, rag_engine, classifier)

    ragas_path = EVAL_DIR / "ood_ragas_dataset.json"
    with open(ragas_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(samples)} RAGAS samples -> {ragas_path}")

    # 3. Evaluate
    logger.info("Evaluating with heuristic RAGAS metrics ...")
    scores = evaluate_ood_samples(samples)

    scores_path = EVAL_DIR / "ood_ragas_scores.json"
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    logger.info(f"Scores saved -> {scores_path}")

    # 4. Build training rows
    if save_training_csv:
        logger.info("Building training CSV rows ...")
        rows = build_training_rows(OOD_CONVERSATIONS, classifier)
        csv_path = EVAL_DIR / "ood_training_rows.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Saved {len(rows)} training rows -> {csv_path}")

    # 5. Print results
    print_sample_preview(samples, n=4)
    print_scores(scores)

    return {
        "conversations": len(OOD_CONVERSATIONS),
        "ragas_samples": len(samples),
        "scores":        scores,
        "paths": {
            "conversations": str(conv_path),
            "ragas_dataset": str(ragas_path),
            "scores":        str(scores_path),
        }
    }


if __name__ == "__main__":
    result = run()
    print("\nDone.")
    print(f"  Conversations : {result['conversations']}")
    print(f"  RAGAS samples : {result['ragas_samples']}")
    print(f"  Output dir    : {EVAL_DIR}/")
