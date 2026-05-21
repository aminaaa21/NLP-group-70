#
# Parsing outside english
#

# import stanza NL
import stanza
stanza.download('nl')
nlp = stanza.Pipeline('nl', processors='tokenize,pos,lemma,depparse')

# write sentences, use full stop to separate them (automatically done in stanza)
doc = nlp("Ik bel hem op. Ik denk dat hij een appel eet.")

# parse
for sent in doc.sentences:
    print("--- new sentence ---")
    for word in sent.words:
        print(f"{word.text} -> head: {sent.words[word.head-1].text if word.head > 0 else 'ROOT'}, relation: {word.deprel}")