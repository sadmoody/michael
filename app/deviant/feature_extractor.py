import stanza
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from .utils import noun_sexiness, verb_sexiness, adjective_sexiness
from .word_collections import SN, BP, PRONOUNS

stanza.download('en')

class FeatureExtractor:

    def __init__(self):
        self.nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma', tokenize_no_ssplit=True, tokenize_batch_size=200, use_gpu=True, verbose=True)
        self.subject_encoder = OneHotEncoder(handle_unknown='ignore')
        self.subject_encoder.fit([[entry] for entry in (list(sorted(PRONOUNS)) + ["OTHER"])])

    def process(self, sents):
        tagged_sents = self.tag(sents)
        processed_sents = np.zeros((len(tagged_sents), 34))
        for i, sent in enumerate(tagged_sents):
            features = self.extract_features(sent)
            processed_sents[i] = self.convert_to_vector(features)
        return(processed_sents)

    def tag(self, sents):
        return self.nlp("\n\n".join(sents)).sentences

    def gen_empty_feature_list(self):
        return {
            'CONTAINS_SN': False, 
            'CONTAINS_BP': False, 
            'CONTAINS_UNSEXY_NOUN': False, 
            'MEAN_NOUN_SEXINESS': 0.0, 
            'CONTAINS_UNSEXY_VERB': False, 
            'MEAN_VERB_SEXINESS': 0.0, 
            'MEAN_ADJECTIVE_SEXINESS': 0.0, 
            'CONTAINS_UNSEXY_ADJECTIVE': False,
            'WORD_COUNT': 0,
            'NOUN_COUNT': 0,
            'VERB_COUNT': 0,
            'ADJECTIVE_COUNT': 0,
            'PROPER_NOUN_COUNT': 0,
            'PRONOUN_COUNT': 0,
            'SUBJECT': "OTHER"}

    def encode_subject(self, subject):
        try:
            return self.subject_encoder.transform(subject).toarray()
        except:
            return None

    def convert_to_vector(self, features):
        feature_length = len(features) - 1
        res = np.zeros((feature_length))
        counter = 0
        for i, feature in enumerate(features):
            if feature == "SUBJECT":
                continue
            res[i] = features[feature]
        subject = self.encode_subject(np.array([[features["SUBJECT"]]]))
        res = np.concatenate((res, subject), axis=None)
        return res

    def extract_features(self, sent):
        features = self.gen_empty_feature_list()
        cum_noun_sexiness = 0.0
        cum_verb_sexiness = 0.0
        cum_adj_sexiness = 0.0
        for word in sent.words:
            features['WORD_COUNT'] += 1
            if word.upos == "NOUN":
                if word.lemma.lower() == "come":
                    word.lemma = "cum"
                if word.lemma.lower() in SN:
                    features['CONTAINS_SN'] = True
                if word.lemma.lower() in BP:
                    features['CONTAINS_BP'] = True
                features['NOUN_COUNT'] += 1
                ns = noun_sexiness(word.lemma.lower())
                cum_noun_sexiness += ns
                if ns <= 10e-7:
                    features['CONTAINS_UNSEXY_NOUN'] = True
            elif word.upos == "VERB":
                if word.lemma.lower() == "come":
                    word.lemma = "cum"
                features['VERB_COUNT'] += 1
                vs = verb_sexiness(word.lemma.lower())
                if word.lemma.lower() == "cum":
                    vs = 0.85
                cum_verb_sexiness += vs
                if vs <= 10e-7:
                    features['CONTAINS_UNSEXY_VERB'] = True
            elif word.upos == "ADJ":
                features['ADJECTIVE_COUNT'] += 1
                adj_sex = adjective_sexiness(word.lemma.lower())
                cum_adj_sexiness += adj_sex
                if adj_sex <= 10e-7:
                    features['CONTAINS_UNSEXY_ADJECTIVE'] = True
            elif word.upos == "PRON":
                features['PRONOUN_COUNT'] += 1
                if word.lemma in PRONOUNS and features['SUBJECT'] == "OTHER":
                    features['SUBJECT'] = word.lemma
            elif word.upos == "PROPN":
                features['PROPER_NOUN_COUNT'] += 1
        if features['NOUN_COUNT'] > 0:
            features['MEAN_NOUN_SEXINESS'] = cum_noun_sexiness / features['NOUN_COUNT']
        if features['VERB_COUNT'] > 0:
            features['MEAN_VERB_SEXINESS'] = cum_verb_sexiness / features['VERB_COUNT']
        if features['ADJECTIVE_COUNT'] > 0:
            features['MEAN_ADJECTIVE_SEXINESS'] = cum_adj_sexiness / features['ADJECTIVE_COUNT']
        return(features)