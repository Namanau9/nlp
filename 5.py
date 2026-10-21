import nltk
import random
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

messages = [
    ("Free entry in 2 a wkly comp to win tickets to the Bahamas!", "spam"),
    ("Congratulations! You have won a $1000 Walmart gift card. Call now.", "spam"),
    ("Hey, are we still meeting for lunch today?", "ham"),
    ("Please call me when you get this message.", "ham"),
    ("Win a brand new car by texting WIN to 12345!", "spam"),
    ("Let's catch up tomorrow for the project discussion.", "ham")
]

stop_words = set(stopwords.words("english"))

def preprocess(message):
    tokens = word_tokenize(message.lower())
    return [
        word for word in tokens
        if word.isalpha() and word not in stop_words
    ]

def extract_features(words):
    return {word: True for word in words}

features = [
    (extract_features(preprocess(msg)), label)
    for msg, label in messages
]

random.shuffle(features)

train_set = features[:4]
test_set = features[4:]

classifier = NaiveBayesClassifier.train(train_set)

accuracy = nltk.classify.accuracy(classifier, test_set)

print("Model Accuracy:", accuracy)

classifier.show_most_informative_features(5)

test_messages = [
    "Claim your free vacation to Hawaii now!",
    "Can you send me the homework assignment?",
    "Win cash prizes by joining the contest today!"
]

print("\nSpam Detection Results:")

for msg in test_messages:
    features = extract_features(preprocess(msg))
    prediction = classifier.classify(features)
    print(f"Message: {msg}\n → Prediction: {prediction}\n")
