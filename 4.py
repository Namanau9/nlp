import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

document = """
Natural Language Processing is one of the most exciting fields in Artificial Intelligence.
It allows computers to understand, interpret, and generate human language.
"""

tokens = word_tokenize(document)

pos_tags = nltk.pos_tag(tokens)

print("Tokens with POS Tags:")

for word, tag in pos_tags:
    print(f"{word:15} --> {tag}")
