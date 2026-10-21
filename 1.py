import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

document = """
Natural Language Processing (NLP) is a field of Artificial Intelligence.
It focuses on the interaction between computers and human languages.
Applications include chatbots, machine translation, and sentiment analysis.
"""

sentences = sent_tokenize(document)

print("Sentences:")
for i, sentence in enumerate(sentences, 1):
    print(f"{i}. {sentence}")

tokens = word_tokenize(document)

print("\nTokens:")
print(tokens)
