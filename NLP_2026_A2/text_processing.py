#
# A2.1 Text Processing & Zipf’s Law (10 points code; 5 points written)
#

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

print('=' * 50)
print('======= BROWN CORPUS =========')
print('=' * 50)

# =========== STEP 1 ===========
def print_frequent_words(input_corpus: list[str], category:str = 'None') -> None:
    if category != 'None':
        input_words = input_corpus.words(categories=category)
    else:
        input_words = input_corpus.words()
    freq = Counter(input_words)
    sorted_words = sorted(list(set(input_words)), key=lambda x: -freq[x])
    print(f'20 most frequent words in category: {category} \n', sorted_words[:20])
    
# list of unique words sorted by descending frequency for 
# (i) the whole corpus
print_frequent_words(brown_corpus)

# (ii) two different genres of your choice.
print_frequent_words(brown_corpus, 'adventure')
print_frequent_words(brown_corpus, 'humor')

print('########')
# =========== STEP 2 ===========
# (i) number of tokens
def print_number_of_tokens(sents):
    sents_list = [' '.join(sent) for sent in sents]
    tokenizer = NLTKWordTokenizer()
    tokenised_text = tokenizer.tokenize_sents(sents_list)
    # flatten to single list
    flatten = list(itertools.chain(*tokenised_text))
    print('Number of tokens in text: ', len(flatten))

brown_sents = brown_corpus.sents()
brown_adventure_sents = brown_corpus.sents(categories='adventure')
brown_humor_sents= brown_corpus.sents(categories='humor')

print_number_of_tokens(brown_sents)
print_number_of_tokens(brown_adventure_sents)
print_number_of_tokens(brown_humor_sents)
print('########')

# (ii) number of types
def print_number_of_types(sents):
    sents_list = [' '.join(sent) for sent in sents]
    tokenizer = NLTKWordTokenizer()
    tokenised_text = tokenizer.tokenize_sents(sents_list)
    # flatten to single list
    # itertool.chain idea from user Shawn Chin on stackoverflow page 'How do I make a flat list out of a list of lists?'
    flatten = list(itertools.chain(*tokenised_text))
    print('Number of types in text: ', len(set(flatten)))

print_number_of_types(brown_sents)
print_number_of_types(brown_adventure_sents)
print_number_of_types(brown_humor_sents)
print('########')

# (iii) number of words
def print_number_of_words(input_corpus: list[str], category:str = 'None') -> None:
    if category != 'None':
        input_words = input_corpus.words(categories=category)
    else:
        input_words = input_corpus.words()
    print(f'number of words in category: {category} \n', len(list(input_words)))

print_number_of_words(brown_corpus)
print_number_of_words(brown_corpus, category='adventure')
print_number_of_words(brown_corpus, category='humor')
print('########')

# (iv) average number of words per sentence
def print_avg_words_per_sentence(sents):
    total_length = 0
    number_of_sents = len(sents)
    for sent in sents:
        total_length += len(sent)
    
    print('Average sent length: ', (total_length/number_of_sents))

print_avg_words_per_sentence(brown_sents)
print_avg_words_per_sentence(brown_humor_sents)
print_avg_words_per_sentence(brown_adventure_sents)
print('########')

# (v) average word length
def print_avg_length_of_words(input_corpus: list[str], category:str = 'None') -> None:
    if category != 'None':
        input_words = input_corpus.words(categories=category)
    else:
        input_words = input_corpus.words()

    total_length = 0
    for word in input_words:
        total_length += len(word)
    avg_length = total_length / len(input_words)
    print(f'average length of words in category {category}:', avg_length)


print_avg_length_of_words(brown_corpus)
print_avg_length_of_words(brown_corpus, category='adventure')
print_avg_length_of_words(brown_corpus, category='humor')
print('########')

#  (vi) number of lemmas
def print_number_of_lemmas(input_corpus, category: str = 'None'):
    if category != 'None':
        input_words = input_corpus.words(categories=category)
    else:
        input_words = input_corpus.words()
    lemmatiser = WordNetLemmatizer()
    lemmas = [lemmatiser.lemmatize(word) for word in input_words]
    print(f'Number of lemmas in category: {category}: ', len(set(lemmas)))

print_number_of_lemmas(brown_corpus)
print_number_of_lemmas(brown_corpus, category='adventure')
print_number_of_lemmas(brown_corpus, category='humor')
print('########')

# =========== STEP 3 ===========
# tagged words

def print_frequent_tag(input_corpus: list[str], category:str = 'None') -> None:
    if category != 'None':
        input_words = input_corpus.tagged_words(categories=category)
    else:
        input_words = input_corpus.tagged_words()

    tags = [word[1] for word in input_words]
    freq = Counter(tags)
    sorted_tags = sorted(list(set(tags)), key=lambda x: -freq[x])
    print(f'\n 10 most frequent tags in category: {category} \n', sorted_tags[:10])

print_frequent_tag(brown_corpus)
print_frequent_tag(brown_corpus, category='adventure')
print_frequent_tag(brown_corpus, category='humor')
print('########')

# =========== STEP 4 ===========

def histogram_frequency_words(input_corpus, category:str='None') -> None:
    if category != 'None':
        input_words = input_corpus.words(categories=category)
    else:
        input_words = input_corpus.words()

    freq = Counter(input_words)
    word_with_freq = freq.most_common()
    sorted_freqs = [freq for _, freq in word_with_freq]

    # regular plot
    plt.plot(range(1, len(sorted_freqs) + 1), sorted_freqs)
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.xlim(0, 1000) # hard to see detail otherwise
    plt.title(f'Zipfs Law - Brown Corpus (cat: {category})')
    plt.show()

    # log-log plot
    plt.plot(range(1, len(sorted_freqs) + 1), sorted_freqs)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Rank (log)')
    plt.ylabel('Frequency (log)')
    plt.title(f'Zipfs Law - Brown Corpus (log-log) (cat: {category})')
    plt.show()

histogram_frequency_words(brown_corpus)
histogram_frequency_words(brown_corpus, category='humor')
histogram_frequency_words(brown_corpus, category='adventure')

# =========== STEP 5 ===========

# Load Indian language corpus
nltk.download('indian')
indian = nltk.corpus.indian
indian_words = indian.words()

# rewriting some functions

def print_frequent_words_indian(input_corpus: list[str], fileid:str = 'None') -> None:
    if fileid != 'None':
        input_words = input_corpus.words(fileids=fileid)
    else:
        input_words = input_corpus.words()
    freq = Counter(input_words)
    sorted_words = sorted(list(set(input_words)), key=lambda x: -freq[x])
    print(f'\n 20 most frequent words in fileid: {fileid} \n', sorted_words[:20])

def print_number_of_words_indian(input_corpus: list[str], fileid:str = 'None') -> None:
    if fileid != 'None':
        input_words = input_corpus.words(fileids=fileid)
    else:
        input_words = input_corpus.words()
    print(f'\n number of words in category: {fileid} \n', len(list(input_words)))

def print_avg_length_of_words_indian(input_corpus: list[str], fileid:str = 'None') -> None:
    if fileid != 'None':
        input_words = input_corpus.words(fileids=fileid)
    else:
        input_words = input_corpus.words()

    total_length = 0
    for word in input_words:
        total_length += len(word)
    avg_length = total_length / len(input_words)
    print(f'\n average length of words in category {fileid}:', avg_length)

def print_number_of_lemmas_indian(input_corpus, fileid: str = 'None'):
    if fileid != 'None':
        input_words = input_corpus.words(fileids=fileid)
    else:
        input_words = input_corpus.words()
    lemmatiser = WordNetLemmatizer()
    lemmas = [lemmatiser.lemmatize(word) for word in input_words]
    print(f'Number of lemmas in category: {fileid}: ', len(set(lemmas)))

def print_frequent_tag_indian(input_corpus: list[str], fileid:str = 'None') -> None:
    if fileid != 'None':
        input_words = input_corpus.tagged_words(fileids=fileid)
    else:
        input_words = input_corpus.tagged_words()

    tags = [word[1] for word in input_words]
    freq = Counter(tags)
    sorted_tags = sorted(list(set(tags)), key=lambda x: -freq[x])
    print(f'\n 10 most frequent tags in category: {fileid} \n', sorted_tags[:10])

def histogram_frequency_words_indian(input_corpus, fileid:str='None') -> None:
    if fileid != 'None':
        input_words = input_corpus.words(fileids=fileid)
    else:
        input_words = input_corpus.words()

    freq = Counter(input_words)
    word_with_freq = freq.most_common()
    sorted_freqs = [freq for _, freq in word_with_freq]

    # regular plot
    plt.plot(range(1, len(sorted_freqs) + 1), sorted_freqs)
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.xlim(0, 1000) # hard to see detail otherwise
    plt.title(f'Zipfs Law - Indian Language Corpus (lang {fileid})')
    plt.show()

    # log-log plot
    plt.plot(range(1, len(sorted_freqs) + 1), sorted_freqs)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Rank (log)')
    plt.ylabel('Frequency (log)')
    plt.title(f'Zipfs Law - Indian Language Corpus (log-log) (lang {fileid})')
    plt.show()


print('=' * 50)
print('======= INDIAN CORPUS =========')
print('=' * 50)

# calling all funcs:
# whole indian corpus
print_frequent_words_indian(indian)
print_number_of_words_indian(indian)
print_avg_length_of_words_indian(indian)
print_number_of_lemmas_indian(indian)
print_frequent_tag_indian(indian)
histogram_frequency_words_indian(indian)
print('########')

# tokens and types (sentences)
indian_sents = indian.sents()
print_number_of_tokens(indian_sents)
print_number_of_types(indian_sents)
print_avg_words_per_sentence(indian_sents)
print('########')

# hindi
print('======= HINDI =========')
print_frequent_words_indian(indian, fileid='hindi.pos')
print_number_of_words_indian(indian, fileid='hindi.pos')
print_avg_length_of_words_indian(indian, fileid='hindi.pos')
print_number_of_lemmas_indian(indian, fileid='hindi.pos')
print_frequent_tag_indian(indian, fileid='hindi.pos')
histogram_frequency_words_indian(indian, fileid='hindi.pos')
print('########')

hindi_sents = indian.sents(fileids='hindi.pos')
print_number_of_tokens(hindi_sents)
print_number_of_types(hindi_sents)
print_avg_words_per_sentence(hindi_sents)
print('########')

# bangla
print('======= BANGLA =========')
print_frequent_words_indian(indian, fileid='bangla.pos')
print_number_of_words_indian(indian, fileid='bangla.pos')
print_avg_length_of_words_indian(indian, fileid='bangla.pos')
print_number_of_lemmas_indian(indian, fileid='bangla.pos')
print_frequent_tag_indian(indian, fileid='bangla.pos')
histogram_frequency_words_indian(indian, fileid='bangla.pos')
print('########')

bangla_sents = indian.sents(fileids='bangla.pos')
print_number_of_tokens(bangla_sents)
print_number_of_types(bangla_sents)
print_avg_words_per_sentence(bangla_sents)
print('########')