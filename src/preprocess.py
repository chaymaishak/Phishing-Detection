import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK data exists
try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))

def simple_clean(text: str) -> str:
    # lower, remove urls, non-alphanum
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_tfidf(corpus, max_features=5000):
    vectorizer = TfidfVectorizer(
        preprocessor=simple_clean,
        stop_words=STOPWORDS,
        max_features=max_features,
        ngram_range=(1,2)
    )
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X
