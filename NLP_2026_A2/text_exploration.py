

# Import packages
import nltk
from nltk import NLTKWordTokenizer, WordNetLemmatizer
import spacy
from collections import Counter
import itertools
import matplotlib.pyplot as plt

# Load Brown language model
nltk.download('brown')
brown_corpus = nltk.corpus.brown
browns_words = nltk.corpus.brown.words()

# =========== EXPLORATION ===========

# print first 50 words to see what tokens look like
print('First 50 words:')
print(list(brown_corpus.words())[:50])

# check if punctuation is included as tokens
print('\nAll unique tokens that are purely punctuation:')
punct_tokens = [w for w in set(brown_corpus.words()) if all(not c.isalpha() and not c.isdigit() for c in w)]
print(punct_tokens)

# check how numbers are handled
print('\nSome number tokens:')
number_tokens = [w for w in set(brown_corpus.words()) if w.isdigit()]
print(list(number_tokens)[:20])

# check capitalisation; find words that appear both capitalised and lowercase
print('\nWords that appear both capitalised and lowercase (first 10):')
all_words = set(brown_corpus.words())
both_cases = [w for w in all_words if w.islower() and w.capitalize() in all_words]
print(both_cases[:10])

# check if \n appears as a token
print('\nDoes \\n appear as a token?')
print('\\n' in set(brown_corpus.words()))

# check how contractions are handled, e.g. don't
print('\nContractions in corpus (first 10):')
contractions = [w for w in set(brown_corpus.words()) if "'" in w]
print(contractions[:10])

# compare token count vs type count vs lemma count
lemmatiser = WordNetLemmatizer()
all_words_list = list(brown_corpus.words())
all_lemmas = [lemmatiser.lemmatize(w) for w in all_words_list]
print('\nToken count:', len(all_words_list))
print('Type count:', len(set(all_words_list)))
print('Lemma count:', len(set(all_lemmas)))