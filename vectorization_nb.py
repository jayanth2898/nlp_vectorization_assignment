import pandas as pd
import time
import nltk
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
df = pd.read_csv("spam.csv", sep="\t", names=["label", "text"])

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Preprocessing
stop_words = set(stopwords.words('english'))

def clean(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df["text"] = df["text"].apply(clean)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# -------- Model --------
model = MultinomialNB()

# -------- BoW --------
start = time.time()
bow = CountVectorizer()
X_train_bow = bow.fit_transform(X_train)
X_test_bow = bow.transform(X_test)

model.fit(X_train_bow, y_train)
pred_bow = model.predict(X_test_bow)

bow_time = time.time() - start

# -------- TF-IDF --------
start = time.time()
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model.fit(X_train_tfidf, y_train)
pred_tfidf = model.predict(X_test_tfidf)

tfidf_time = time.time() - start

# -------- N-GRAM (NEW PART) --------
start = time.time()
ngram = CountVectorizer(ngram_range=(1,2))
X_train_ng = ngram.fit_transform(X_train)
X_test_ng = ngram.transform(X_test)

model.fit(X_train_ng, y_train)
pred_ng = model.predict(X_test_ng)

ng_time = time.time() - start

# -------- Results --------
def evaluate(y, pred):
    return accuracy_score(y, pred), f1_score(y, pred)

print("BoW:", evaluate(y_test, pred_bow), "Time:", bow_time)
print("TF-IDF:", evaluate(y_test, pred_tfidf), "Time:", tfidf_time)
print("N-gram:", evaluate(y_test, pred_ng), "Time:", ng_time)