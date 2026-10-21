import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os

nltk.data.path.append(os.path.join(os.path.expanduser('~'), 'nltk_data'))

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

document = """
The striped bats are hanging on their feet for best.
Cats running faster than dogs.
He studies in the university and was studying yesterday too.
"""

tokens = word_tokenize(document)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_words = [stemmer.stem(word) for word in tokens]

lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]

print("Original Tokens:")
print(tokens)

print("\nStemmed Words:")
print(stemmed_words)

print("\nLemmatized Words:")
print(lemmatized_words)
