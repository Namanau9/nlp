import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = [
    ("Deep learning is transforming AI.", "Technology"),
    ("10 tips for a healthy lifestyle.", "Health"),
    ("Exploring the mountains of Switzerland.", "Travel"),
    ("Quantum computing and future of tech.", "Technology"),
    ("Yoga improves mental health.", "Health"),
    ("Trip to Japan: temples and sushi.", "Travel"),
    ("AI in healthcare: predictive models.", "Technology"),
]

df = pd.DataFrame(
    data,
    columns=["text", "label"]
)

tfidf = TfidfVectorizer(
    stop_words="english"
)

X = tfidf.fit_transform(df["text"])

y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

clf = LogisticRegression(
    max_iter=1000
)

clf.fit(X_train, y_train)

def predict_blog(text):
    vec = tfidf.transform([text])
    return clf.predict(vec)[0]

samples = [
    "AI is helping doctors detect cancer early.",
    "I enjoyed hiking in the Himalayas.",
    "Best exercises for fitness and meditation."
]

for s in samples:
    print(
        f"Blog: {s}\n"
        f" → Label: {predict_blog(s)}\n"
    )
