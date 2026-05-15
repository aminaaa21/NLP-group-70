#
#
#


import stanza

# English
stanza.download('en')

# setup
nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse')

# gp sentence
doc = nlp("The horse raced past the barn fell.")

# print dependencies
for sent in doc.sentences:
    for word in sent.words:
        print(f"{word.text} -> head: {sent.words[word.head-1].text if word.head > 0 else 'ROOT'}, relation: {word.deprel}")