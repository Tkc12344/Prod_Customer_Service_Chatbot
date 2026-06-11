import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
import joblib
import os
from nltk.stem import PorterStemmer
import nltk
import re

# Download nltk data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# ---------------------------------------------------------------------------
# Extra utterances for historically weak intents — injected at train time
# ---------------------------------------------------------------------------
AUGMENT_DATA = {
    'goodbye': [
        'bye', 'goodbye', 'see you', 'see ya', 'take care', 'farewell',
        'talk later', 'catch you later', 'have a good day', 'have a great day',
        'thanks bye', 'ok bye', 'alright bye', 'cheers bye', 'ciao',
        'until next time', 'all done', 'all sorted', 'nothing further',
        'that will be all', 'i am done', 'we are done here', 'done for now',
        'no more questions', 'no further questions', 'thats all i needed',
        'thats everything', 'i think thats it', 'i am good now',
        'everything is sorted', 'issue resolved', 'problem solved',
        'all good now', 'im all set', 'im good thanks', 'thanks thats all',
        'thank you and goodbye', 'thanks and bye', 'ok thanks bye',
        'great thanks bye', 'perfect thanks bye', 'wonderful thanks bye',
        'until we meet again', 'speak soon', 'talk soon', 'later',
        'good night', 'good evening', 'have a nice day', 'have a good one',
        'signing off', 'logging off', 'closing chat', 'end chat',
        'close this chat', 'end conversation', 'finish', 'done',
        'nothing else thanks', 'nothing else needed', 'no other questions',
        'i think im done', 'i think thats everything', 'thats it from me',
    ],
    'greeting': [
        'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
        'hi there', 'hey there', 'howdy', 'greetings', 'whats up', 'sup',
        'hello there', 'hi i need help', 'hey i have a question',
        'hello i am a new customer', 'hi i am a returning customer',
        'good day', 'morning', 'afternoon', 'evening',
        'hi can you help me', 'hello can you assist me',
        'hey i need assistance', 'hi i need support',
        'hello i have an issue', 'hi i have a problem',
        'hey i have a concern', 'hello i have a query',
        'hi i have a question', 'hello i need help',
        'hey can i ask something', 'hi is anyone there',
        'hello is this support', 'hi this is support right',
        'hey i want to chat', 'hello i want to talk to someone',
        'hi i just wanted to ask', 'hello just checking in',
        'hey just a quick question', 'hi quick question',
        'hello quick query', 'hey quick query',
        'what do you do', 'what can you help with',
        'what can you do', 'how can you help me',
        'who are you', 'are you a bot', 'am i talking to a bot',
        'is this automated', 'is this a real person',
        'start', 'begin', 'help', 'i need help',
    ],
    'thanks': [
        'thank you', 'thanks', 'thank you so much', 'thanks a lot',
        'many thanks', 'much appreciated', 'appreciate it', 'appreciate your help',
        'thanks for your help', 'thank you for your help',
        'thanks for the help', 'thank you for the assistance',
        'thanks for sorting that', 'thank you for sorting that out',
        'thanks for resolving', 'thank you for resolving that',
        'great help', 'very helpful', 'you were very helpful',
        'you were so helpful', 'that was helpful', 'that was very helpful',
        'this was helpful', 'this was very helpful',
        'you are amazing', 'you are great', 'you are awesome',
        'brilliant help', 'excellent help', 'fantastic help',
        'wonderful help', 'superb help', 'outstanding help',
        'five star service', 'five stars', 'top notch service',
        'great service', 'excellent service', 'fantastic service',
        'wonderful service', 'superb service', 'outstanding service',
        'you helped me a lot', 'you really helped me',
        'that really helped', 'that solved my problem',
        'problem solved thanks', 'issue resolved thanks',
        'all sorted thanks', 'all good thanks',
        'perfect thanks', 'great thanks', 'brilliant thanks',
        'cheers', 'cheers mate', 'ta', 'ta very much',
        'grateful', 'so grateful', 'very grateful',
        'a thousand thanks', 'a million thanks',
        'this was easy thanks', 'that was quick thanks',
        'thank u', 'thx', 'ty', 'tysm', 'thnx',
        'i really appreciate it', 'i really appreciate your help',
        'i am grateful for your help', 'i am thankful',
        'issue is fixed thanks', 'all done thanks',
    ],
    'human_agent': [
        'speak to a human', 'talk to a human', 'speak to a person',
        'talk to a person', 'speak to an agent', 'talk to an agent',
        'speak to a real person', 'talk to a real person',
        'connect me to an agent', 'connect me to a human',
        'transfer me to an agent', 'transfer to agent',
        'i want a human', 'i need a human', 'i want a real person',
        'i need a real person', 'i want to speak to someone',
        'i need to speak to someone', 'i want to talk to someone',
        'i need to talk to someone', 'get me a human',
        'get me an agent', 'get me a real person',
        'human please', 'agent please', 'real person please',
        'live agent', 'live chat', 'live support',
        'live agent please', 'live chat please', 'live support please',
        'connect to live agent', 'connect to live chat',
        'i want live support', 'i need live support',
        'escalate this', 'escalate my issue', 'i need this escalated',
        'please escalate', 'escalate to a manager', 'speak to a manager',
        'talk to a manager', 'i want to speak to a manager',
        'i need to speak to a manager', 'get me a manager',
        'manager please', 'supervisor please', 'speak to supervisor',
        'talk to supervisor', 'i want a supervisor',
        'call center', 'phone support', 'call support',
        'i want to call', 'i need to call', 'phone number',
        'contact support', 'contact a human', 'contact an agent',
        'this is urgent', 'this is an emergency', 'urgent help needed',
        'i need urgent help', 'i need immediate help',
        'this needs immediate attention', 'please help me now',
        'i am very frustrated', 'i am very upset', 'i am very angry',
        'this is unacceptable', 'i demand to speak to someone',
        'i insist on speaking to a human', 'bot is not helping',
        'you are not helping', 'this bot is useless',
        'i want human assistance', 'human assistance please',
        'can i speak to a real agent', 'can i talk to a real agent',
        'is there a human i can talk to', 'is there a person i can talk to',
    ],
    'feedback': [
        'i want to leave feedback', 'i want to give feedback',
        'i have feedback', 'i have a complaint', 'i want to complain',
        'i have a compliment', 'i want to compliment',
        'i have a suggestion', 'i have an idea', 'i have a recommendation',
        'i want to report a problem', 'i want to report an issue',
        'i want to report a bug', 'report a problem', 'report an issue',
        'submit feedback', 'submit a complaint', 'submit a review',
        'leave a review', 'write a review', 'post a review',
        'rate the service', 'rate my experience', 'rate this interaction',
        'share my experience', 'share my feedback', 'share my thoughts',
        'my experience was great', 'my experience was terrible',
        'my experience was good', 'my experience was bad',
        'my experience was poor', 'my experience was excellent',
        'overall experience', 'overall feedback', 'overall rating',
        'i am satisfied', 'i am not satisfied', 'i am very satisfied',
        'i am very dissatisfied', 'i am happy with the service',
        'i am unhappy with the service', 'i am disappointed',
        'i am very disappointed', 'i am impressed',
        'great experience', 'terrible experience', 'poor experience',
        'excellent experience', 'bad experience', 'good experience',
        'how to submit a complaint', 'how do i complain',
        'where do i leave feedback', 'how do i leave a review',
        'i want to praise your service', 'i want to criticize your service',
        'constructive feedback', 'positive feedback', 'negative feedback',
        'customer satisfaction', 'service quality feedback',
        'i have a comment', 'i have comments', 'i want to comment',
        'i want to share my opinion', 'i have an opinion',
        'this service needs improvement', 'you need to improve',
        'i think you should', 'i suggest you', 'my suggestion is',
        'feature request', 'i would like to request a feature',
        'can you add a feature', 'can you improve',
        'report a problem', 'report a bug', 'report an error',
        'i found a bug', 'i found an error', 'i found a problem',
    ],
    'out_of_scope': [
        'what is the weather', 'weather today', 'weather forecast',
        'tell me a joke', 'make me laugh', 'say something funny',
        'what is the news', 'latest news', 'news today',
        'what is the stock price', 'best stocks to buy', 'stock market',
        'recommend a book', 'best book to read', 'book recommendation',
        'recommend a movie', 'best movie to watch', 'movie recommendation',
        'recommend a restaurant', 'best restaurant near me',
        'what is the capital of france', 'geography question',
        'what is the meaning of life', 'philosophical question',
        'tell me about history', 'history question',
        'what is artificial intelligence', 'explain machine learning',
        'what is the population of', 'population question',
        'translate this for me', 'translate to spanish',
        'what language is this', 'language question',
        'write me a poem', 'write me a story', 'write me an essay',
        'do my homework', 'help me with homework',
        'what is the time', 'what time is it', 'current time',
        'what is todays date', 'what day is it today',
        'play music', 'play a song', 'music recommendation',
        'sports scores', 'who won the game', 'sports news',
        'recipe for', 'how to cook', 'cooking tips',
        'medical advice', 'health advice', 'doctor recommendation',
        'legal advice', 'lawyer recommendation', 'legal question',
        'financial advice', 'investment advice', 'crypto advice',
        'dating advice', 'relationship advice', 'personal advice',
        'political opinion', 'politics question', 'election results',
        'religious question', 'spiritual advice', 'prayer',
        'math problem', 'solve this equation', 'calculate',
        'science question', 'physics question', 'chemistry question',
        'coding help', 'programming question', 'debug my code',
        'what is your name', 'who made you', 'who created you',
        'are you sentient', 'do you have feelings', 'are you conscious',
        'what is your favorite color', 'do you have a favorite',
        'can you be my friend', 'will you marry me',
        'tell me something interesting', 'random fact',
        'what is bitcoin', 'explain blockchain', 'nft question',
        'social media help', 'instagram help', 'twitter help',
        'gaming question', 'video game help', 'game recommendation',
        'travel advice', 'best place to visit', 'travel tips',
        'fitness advice', 'workout tips', 'diet advice',
    ],
}


class IntentClassifier:
    def __init__(self):
        self.stemmer = PorterStemmer()
        # Word-level TF-IDF (1-3 grams)
        self.word_vectorizer = TfidfVectorizer(
            max_features=8000,
            stop_words=None,
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True,
            strip_accents='unicode',
        )
        # Character-level TF-IDF — catches typos and morphological variants
        self.char_vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            max_features=5000,
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
            strip_accents='unicode',
        )
        # Calibrated LinearSVC — fast, strong on text, gives probabilities
        base_svc = LinearSVC(
            C=0.8,
            max_iter=2000,
            class_weight='balanced',
            random_state=42,
        )
        self.model = CalibratedClassifierCV(base_svc, cv=5, method='sigmoid')
        self.intents = None
        self._word_matrix = None
        self._char_matrix = None

    # ------------------------------------------------------------------
    def _clean(self, text: str) -> str:
        """Lowercase, remove punctuation noise, collapse whitespace."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s']", ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def stem_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ''
        return ' '.join(self.stemmer.stem(w) for w in text.split())

    def preprocess(self, text: str) -> str:
        return self.stem_text(self._clean(text))

    # ------------------------------------------------------------------
    def load_data(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f'Dataset not found: {filepath}')
        df = pd.read_csv(filepath, encoding='latin-1')
        if 'utterance' not in df.columns or 'intent' not in df.columns:
            raise ValueError("CSV must have 'utterance' and 'intent' columns")
        X = df['utterance'].fillna('').astype(str).tolist()
        y = df['intent'].astype(str).tolist()

        # Inject augmentation data for weak intents
        for intent, utterances in AUGMENT_DATA.items():
            for utt in utterances:
                X.append(utt)
                y.append(intent)

        X = [self.preprocess(t) for t in X]
        self.intents = np.unique(y)
        print(f"Loaded {len(X)} samples across {len(self.intents)} intents")
        # Print per-intent counts
        from collections import Counter
        counts = Counter(y)
        for intent in sorted(counts):
            print(f"  {intent}: {counts[intent]}")
        return X, y

    # ------------------------------------------------------------------
    def _build_features(self, X, fit=False):
        """Combine word and char TF-IDF into a single sparse matrix."""
        from scipy.sparse import hstack
        if fit:
            W = self.word_vectorizer.fit_transform(X)
            C = self.char_vectorizer.fit_transform(X)
        else:
            W = self.word_vectorizer.transform(X)
            C = self.char_vectorizer.transform(X)
        return hstack([W, C])

    # ------------------------------------------------------------------
    def train(self, X, y):
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from scipy.sparse import hstack

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )

        print("Building feature matrices …")
        X_train_feat = self._build_features(X_train, fit=True)
        X_test_feat  = self._build_features(X_test,  fit=False)

        print("Training model …")
        self.model.fit(X_train_feat, y_train)

        # Cross-validation on training set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.model, X_train_feat, y_train, cv=cv, scoring='accuracy'
        )
        print(f'CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})')

        # Hold-out evaluation
        y_pred = self.model.predict(X_test_feat)
        print('\nModel Evaluation (hold-out):')
        print(classification_report(y_test, y_pred))

    # ------------------------------------------------------------------
    def save_model(self, filepath='intent_model.pkl'):
        joblib.dump(
            (self.word_vectorizer, self.char_vectorizer, self.model, self.intents),
            filepath
        )
        print(f'Model saved to {filepath}')

    def load_model(self, filepath='intent_model.pkl'):
        if not os.path.exists(filepath):
            print(f'Model not found: {filepath}')
            return False
        loaded = joblib.load(filepath)
        if len(loaded) == 4:
            # New format: word_vec, char_vec, model, intents
            self.word_vectorizer, self.char_vectorizer, self.model, self.intents = loaded
        elif len(loaded) == 3:
            # Legacy format: vectorizer, model, intents — patch in a dummy char vectorizer
            print("Legacy model format detected — char vectorizer unavailable, retraining recommended.")
            self.word_vectorizer, self.model, self.intents = loaded
            self._legacy_mode = True
        print(f'Model loaded from {filepath}')
        return True

    def _legacy_predict(self, utterance: str):
        """Fallback predict for legacy (3-tuple) models without char vectorizer."""
        processed = self.preprocess(utterance)
        vec = self.word_vectorizer.transform([processed])
        proba = self.model.predict_proba(vec)[0]
        return self.model.predict(vec)[0], float(np.max(proba))

    def predict(self, utterance: str):
        """Predict intent — handles both new and legacy model formats."""
        if not isinstance(utterance, str) or not utterance.strip():
            return 'other', 0.0
        try:
            if getattr(self, '_legacy_mode', False):
                return self._legacy_predict(utterance)
            processed = self.preprocess(utterance)
            feat = self._build_features([processed], fit=False)
            proba = self.model.predict_proba(feat)[0]
            confidence = float(np.max(proba))
            intent = self.model.predict(feat)[0]
            return intent, confidence
        except Exception:
            return 'other', 0.0


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time
    timestamp = int(time.time())
    classifier = IntentClassifier()
    model_file = f'intent_model_v2_{timestamp}.pkl'
    print("Training new model from intents_enhanced_2.csv …")
    X, y = classifier.load_data('intents_enhanced_2.csv')
    classifier.train(X, y)
    classifier.save_model(model_file)
    print(f"\nNew model saved as {model_file}")
    print(f"\nUpdate .env:  MODEL_PATH={model_file}")
