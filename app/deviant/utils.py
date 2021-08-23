from .word_collections import NS, VS, AS
from nltk.tag.util import untag, str2tuple
import re

def noun_sexiness(noun):
    if noun in NS:
        return NS[noun]
    return 10e-7

def verb_sexiness(verb):
    if verb in VS:
        return VS[verb]
    return 10e-7

def adjective_sexiness(adj):
    if adj in AS:
        return AS[adj]
    return 10e-7

def untag_file(blob):
    res = []
    for line in blob.split('\n'):
        sent = line.strip('\n').split(' ')
        tagged_sent = [str2tuple(pair) for pair in sent]
        untagged_sent = untag(tagged_sent)
        joined_sent = []
        for word in untagged_sent:
            if (len(joined_sent) > 0):
                if not re.match("^[a-zA-Z0-9_]*$", word):
                    joined_sent[len(joined_sent)-1] += word
                    continue
            joined_sent.append(word)
        res.append(" ".join(joined_sent))
    return res