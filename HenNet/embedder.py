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

    def read_processed_data(self):
        path = os.getcwd() + '/data/processed'
        print('Reading json file ...')
        with open(path + '/contexts_dev.json') as f:
            context = json.load(f)
        with open(path + '/questions_dev.json') as f:
            questions = json.load(f)
        with open(path + '/responses_dev.json') as f:
            responses = json.load(f)
        return self.sanity_check(context, questions, responses)

    def load_embeddings(self):
        path = os.getcwd() + '/data/bert/'
        with open(path + 'context_dev.pickle', 'rb') as f:
            context_emb = pickle.load(f)
        with open(path + 'questions_dev.pickle', 'rb') as f:
            questions_emb = pickle.load(f)
        with open(path + 'responses_dev.pickle', 'rb') as f:
            responses_emb = pickle.load(f)
        return (context_emb, questions_emb, responses_emb)

    def sanity_check(self, context, questions, responses):
        # context sanity
        for k, v in context.items():
            sents, l = self.extract_sentences(v['sentences'], v['word']), 0
            for s in sents:
                l += len(s)
            assert len(v['ent']) == len(v['lemma']) == len(v['pos']) == len(v['word']) == l
        # questions sanity
        for k, v in questions.items():
            sents, l = self.extract_sentences(v['sentences'], v['word']), 0
            for s in sents:
                l += len(s)
            assert len(v['ent']) == len(v['lemma']) == len(v['pos']) == len(v['word']) == l
        # response sanity
        for k, v in responses.items():
            sents, l = self.extract_sentences(v['sentences'], v['word']), 0
            for s in sents:
                l += len(s)
            assert len(v['ent']) == len(v['lemma']) == len(v['pos']) == len(v['word']) == l
        assert len(questions) == len(responses)
        print ('All sanity checking passed !')
        return (context, questions, responses)

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
                    subword_encoding, subword_tokens = bc.encode([original], is_tokenized=False, show_tokens=True)
                    sub_dim = len(subword_encoding[0])-1
                    subword_encoding, subword_tokens = subword_encoding[0][1:sub_dim], subword_tokens[0][1:sub_dim]
                    subword_vector = np.average(subword_encoding, axis=0)
                    full_tokens.append(tokens[i])
                    full_vectors.append(subword_vector)
                else:
                    full_tokens.append(tokens[i])
                    full_vectors.append(vectors[i])
            continue

        assert len(full_vectors) == len(full_tokens) == len(original_tokens)
        full_vectors = np.array(full_vectors)
        return full_vectors

    def process_CoQA(self):
        context, questions, responses = self.read_processed_data()
        context_emb, questions_emb, responses_emb = {}, {}, {}

        # generate bert embeddings
        print ('generating context vectors ...')
        time.sleep(1.0)
        for k, v in tqdm(context.items()):
            sents = self.extract_sentences(v['sentences'], v['lemma'])
            bert_v = self.get_contextual_embeddings(sents, v['lemma'])
            bert_v = self.sum_4_layers(bert_v)
            context_emb[k] = bert_v
            if len(context_emb) == 5:
                break

        print('generating question vectors ...')
        time.sleep(1.0)
        for k, v in tqdm(questions.items()):
            sents = self.extract_sentences(v['sentences'], v['lemma'])
            bert_v = self.get_contextual_embeddings(sents, v['lemma'])
            bert_v = self.sum_4_layers(bert_v)
            questions_emb[k] = bert_v
            if len(questions_emb) == 5:
                break

        print ('generating response vectors ...')
        time.sleep(1.0)
        for k, v in tqdm(responses.items()):
            sents = self.extract_sentences(v['sentences'], v['lemma'])
            bert_v = self.get_contextual_embeddings(sents, v['lemma'])
            bert_v = self.sum_4_layers(bert_v)
            responses_emb[k] = bert_v
            if len(responses_emb) == 5:
                break

        print ('saving vectors into files ...')
        time.sleep(1.0)
        path = os.getcwd() + '/data/bert/'
        with open(path + 'context_dev.pickle', 'wb') as f:
            pickle.dump(context_emb, f, protocol=4)
        with open(path + 'questions_dev.pickle', 'wb') as f:
            pickle.dump(questions_emb, f, protocol=4)
        with open(path + 'responses_dev.pickle', 'wb') as f:
            pickle.dump(responses_emb, f, protocol=4)
        print ('embedder completed !\n')
        return

    def sum_4_layers(self, vectors):
        summed = []
        for v in vectors:
            v1, v2, v3, v4 = v[0:1024], v[1024:2048], v[2048:3072], v[3072:4096]
            sum_v = v1 + v2 + v3 + v4
            summed.append(sum_v)
        summed = np.array(summed)
        return summed

    def extract_sentences(self, sentences, words):
        sents = []
        for start, end in sentences:
            sents.append(words[start:end])
        return sents


emb = CoQAEmbeddor()
emb.process_CoQA()
# a, b, c = emb.load_embeddings()
