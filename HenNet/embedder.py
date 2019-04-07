# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm
from bert_serving.client import BertClient
from bert_serving.server import BertServer
import os, json, pickle, time

class CoQAEmbeddor():
    def __init__(self):
        self.context_data, self.embeddings = {}, {}
        self.data_path = os.getcwd() + '/data/processed/'
        self.embedding_path = os.getcwd() + '/embeddings/'
        self.visited = {}

    def get_contextual_embeddings(self, sents, original_tokens):
        bc = BertClient()

        full_vectors, full_tokens = [], []
        for sent in sents:
            encoding = bc.encode([sent], is_tokenized=True, show_tokens=True)
            vectors, tokens = np.squeeze(encoding[0]), np.squeeze(np.array(encoding[1]))
            en_dim = len(vectors)-1
            vectors, tokens = vectors[1:en_dim], tokens[1:en_dim]
            for i in range(len(tokens)):
                if tokens[i] == '[UNK]':
                    original = sent[i]
                    if original in ['<Q-END>','<A-END>']:
                        full_vectors.append(np.zeros(shape=(vectors.shape[1])))
                        full_tokens.append(tokens[i])
                    else:
                        subword_encoding, subword_tokens = bc.encode([original], is_tokenized=False, show_tokens=True)
                        sub_dim = len(subword_encoding[0])-1
                        subword_encoding, subword_tokens = subword_encoding[0][1:sub_dim], subword_tokens[0][1:sub_dim]
                        subword_vector = np.average(subword_encoding, axis=0)
                        full_vectors.append(subword_vector)
                        full_tokens.append(tokens[i])
                else:
                    full_vectors.append(vectors[i])
                    full_tokens.append(tokens[i])
            continue

        # print (len(full_vectors), len(full_tokens), len(original_tokens))
        assert len(full_vectors) == len(full_tokens) == len(original_tokens)
        return np.array(full_vectors)

    def process_CoQA(self, option='dev', save=False, conv_limit=999999):
        # read data
        path = os.getcwd() + '/data/coqa-{}-preprocessed.json'.format(option)
        with open(path) as f:
            print('Reading json file ...')
            j = json.load(f)
            num_conv = len(j['data'])
            print ('Read {} conversations\n'.format(num_conv))

        count = 0
        # process each conversation
        print ('preparing BERT embedding ...')
        for x in tqdm(j['data'], total=num_conv):
            if count == conv_limit:
                break

            # context preprocessing
            original_context, annotated_context, context_id = x['context'], x['annotated_context'], x['id']
            words, sents  = annotated_context['lemma'], annotated_context['sentences']
            sents = self.extract_sentences(sents, words)
            # context_embedding = self.get_contextual_embeddings(sents, words)

            # history parsing for each turn
            for history in x['qas']:
                q, a = history['annotated_question'], history['annotated_answer']
                q_word, q_lemma = q['word'], q['lemma']
                a_word, a_lemma = a['word'], a['lemma']
                continue



    def extract_sentences(self, sentences, words):
        sents = []
        for start, end in sentences:
            sents.append(words[start:end])
        return sents


emb = CoQAEmbeddor()
emb.process_CoQA()