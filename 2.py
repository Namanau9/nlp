import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

document = """
Natural Language Processing is amazing. However, sometimes we need to filter out
unnecessary words.
This helps in better processing and understanding of language.
"""

stop_words = set(stopwords.words('english'))

custom_stop_words = {'however', 'sometimes', 'need'}

all_stop_words = stop_words.union(custom_stop_words)

tokens = word_tokenize(document)

filtered_tokens = [
    word for word in tokens
    if word.lower() not in all_stop_words and word.isalpha()
]

print("Filtered Tokens (without stop words):")
print(filtered_tokens)
