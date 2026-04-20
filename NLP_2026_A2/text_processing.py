#
# A2.1 Text Processing & Zipf’s Law (10 points code; 5 points written)
#

# Import packages
import nltk
import spacy
from collections import Counter


# Load Brown language model
# nlp = spacy.load('nltk.corpus.brown')

nltk.download('brown')
brown_corpus = nltk.corpus.brown
browns_words = nltk.corpus.brown.words()

def print_frequent_words(input_corpus: list[str], category:str = 'None') -> None:
    if category != 'None':
        input_words = input_corpus.words(categories=category)
    else:
        input_words = input_corpus.words()
    freq = Counter(input_words)
    sorted_words = sorted(list(set(input_words)), key=lambda x: -freq[x])
    print(f'\n\n 20 most frequent words in category: {category} \n', sorted_words[:20])
    

# list of unique words sorted by descending frequency for (i) the whole corpus
print_frequent_words(brown_corpus)

# list of unique words sorted by descending frequency for (ii) two different
# genres of your choice.
print_frequent_words(brown_corpus, 'adventure')
print_frequent_words(brown_corpus, 'humor')


