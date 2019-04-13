# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from collections import Counter
import os, json, pickle, time, copy

class CoQAPreprocessor():
    def __init__(self, option, c_pad, h_pad, bert_path):
        self.option = option
        self.c_pad, self.h_pad = c_pad, h_pad
        self.context_len, self.history_len, self.exclude = 0, 0, []

        self.pos_map = {'WP$': 40, 'IN': 10, 'WDT': 14, 'PRP': 25, 'TO': 22, 'LS': 47, 'JJ': 5, 'CD': 17, 'VBP': 30, 'DT': 4,
         'NNS': 21, 'NFP': 44, 'WRB': 24, 'VB': 23, '_SP': 51, 'ADD': 49, 'NN': 7, 'NNPS': 43, 'XX': 46, 'NNP': 2,
         'RBR': 34, "''": 26, '$': 45, ',': 11, '-RRB-': 9, '-LRB-': 8, 'MD': 35, 'SYM': 41, 'PDT': 38, 'VBD': 16,
         'END': 1, 'PRP$': 13, 'FW': 48, 'CC': 20, ':': 18, 'PAD': 0, 'RB': 15, 'VBZ': 3, 'AFX': 50, 'POS': 39,
         '``': 29, 'VBN': 19, 'JJR': 31, 'RP': 27, 'VBG': 28, 'JJS': 37, '.': 12, 'RBS': 32, 'HYPH': 6, 'UH': 42,
         'WP': 33, 'EX': 36}
        self.ent_map = {'END': 1, 'I-CARDINAL': 25, 'B-QUANTITY': 22, 'B-CARDINAL': 10, 'I-NORP': 34, 'B-EVENT': 26, 'B-ORG': 7,
         'I-MONEY': 21, 'I-PERCENT': 33, 'I-ORG': 11, 'B-NORP': 12, 'B-ORDINAL': 24, 'I-WORK_OF_ART': 19, 'B-PRODUCT': 13,
         'I-TIME': 17, 'B-TIME': 16, 'B-LANGUAGE': 31, 'I-EVENT': 27, 'I-DATE': 9, 'I-QUANTITY': 23, 'I-PRODUCT': 35,
         'B-DATE': 8, 'B-WORK_OF_ART': 18, 'O': 4, 'I-PERSON': 3, 'I-GPE': 6, 'I-FAC': 36, 'I-LOC': 15, 'I-LAW': 30,
         'B-GPE': 5, 'B-LOC': 14, 'B-LAW': 29, 'B-MONEY': 20, 'B-FAC': 28, 'B-PERCENT': 32, 'B-PERSON': 2, 'PAD': 0}

        self.data_path = os.getcwd() + '/data/processed/'
        self.bert_path = bert_path
        self.context_emb, self.questions_emb, self.responses_emb = self.load_embeddings()

    def load_embeddings(self):
        option, path = self.option, self.bert_path
        with open(path + 'context_{}.pickle'.format(option), 'rb') as f:
            context_emb = pickle.load(f)
        with open(path + 'questions_{}.pickle'.format(option), 'rb') as f:
            questions_emb = pickle.load(f)
        with open(path + 'responses_{}.pickle'.format(option), 'rb') as f:
            responses_emb = pickle.load(f)
        return (context_emb, questions_emb, responses_emb)

    def load_processed_data(self):
        option = self.option
        with open(self.data_path + 'contexts_{}.json'.format(option)) as f:
            c = json.load(f)
        with open(self.data_path + 'questions_{}.json'.format(option)) as f:
            q = json.load(f)
        with open(self.data_path + 'responses_{}.json'.format(option)) as f:
            r = json.load(f)
        assert len(q) == len(r)

        for cid, v in c.items():
            v['history'], v['contextual_history'] = [], []
            num_words = len(v['word'])
            if self.context_len < num_words:
                self.context_len = num_words
            if num_words > self.c_pad:
                self.exclude.append(cid)

        for qid, v in q.items():
            cid = qid.split('_')[0]
            c[cid]['history'].append(qid)

        # exclude long sessions
        for cid in self.exclude:
            del c[cid]

        # convert pos and ent maps to categorical labels
        self.pos_map = self.categorical_dict(self.pos_map)
        self.ent_map = self.categorical_dict(self.ent_map)

        print ('{} {} conversations loaded'.format(len(c), option))
        return (c, q, r)

    def categorical_dict(self, map):
        values = list(map.values())
        values = to_categorical(values)
        index = 0
        for k, v in map.items():
            map[k] = values[index]
            index += 1
        return map

    def join_by_id(self, context, questions, responses, window=3):
        # join data
        for context_id, v in context.items():
            if len(v['history']) <= 1:
                continue

            history = v['history']
            contextual_samples = []
            for i in range(len(history) - window + 1):
                contextual_samples.append(history[i:i+window])
            for i in range(window-1):
                pad = history[0:i+1]
                dummy = [None] * (window - len(pad))
                pad = dummy + pad
                contextual_samples.insert(i, pad)
            v['contextual_history'] = contextual_samples
            assert(len(v['contextual_history']) == len(v['history']))

        # prepare training entries
        train, test = [], []
        for context_id, v in context.items():
            for h in v['contextual_history']:
                entry = (context_id, h)
                train.append(entry)
                test.append(responses[h[-1]]['answer_span'])
        return (train, test)

    def prepare_training(self, limit=500):
        c_pad, h_pad = self.c_pad, self.h_pad
        context, questions, responses = self.load_processed_data()
        train, test = self.join_by_id(context, questions, responses, window=3)
        context_emb, questions_emb, responses_emb = self.context_emb, self.questions_emb, self.responses_emb

        test = context_emb['36w0ob37hwe5i7eo0mew1h7lpagzhc']

        # fill up values
        context_map, targets = {}, []
        h_emb, h_pos, h_ent, cids, tids = [], [], [], [], []
        iteration = min(limit, len(train))
        for i in tqdm(range(iteration)):
            cid, history, tid = train[i][0], train[i][1], train[i][1][-1]
            history_inputs = {'context_id': cid, 'history': [], 'history_pos': [], 'history_ent': []}
            context_inputs = {'context': [], 'context_pos': [], 'context_ent': []}

            # context input
            context_inputs['context'] = context_emb[cid]
            context_inputs['context_pos'] = np.array(context[cid]['pos'], dtype=object)
            context_inputs['context_ent'] = np.array(context[cid]['ent'], dtype=object)

            # context padding
            context_inputs['context'] = pad_sequences(context_inputs['context'].T, maxlen=c_pad, dtype=float, value=0.0).T
            context_inputs['context_pos'] = list(pad_sequences(np.expand_dims(context_inputs['context_pos'], axis=0), maxlen=c_pad, dtype=object, value='PAD')[0])
            context_inputs['context_ent'] = list(pad_sequences(np.expand_dims(context_inputs['context_ent'], axis=0), maxlen=c_pad, dtype=object, value='PAD')[0])

            # one hot
            for x in range(len(context_inputs['context_pos'])):
                p, e = context_inputs['context_pos'][x], context_inputs['context_ent'][x]
                context_inputs['context_pos'][x] = self.pos_map[p]
                context_inputs['context_ent'][x] = self.ent_map[e]

            # history inputs
            prev, current = history[0:len(history)-1], history[-1]
            history, history_pos, history_ent, span = self.generate_history_sequence(prev, current, questions, responses, h_pad)
            history_inputs['history'], history_inputs['history_pos'], history_inputs['history_ent'] = history, history_pos, history_ent

            # span padding
            start, end = span[0], span[1]
            span_expanded = np.zeros(shape=(2, len(context[cid]['word'])))
            span_expanded[0][start] = 1.0
            span_expanded[1][end] = 1.0
            span_expanded = pad_sequences(span_expanded, maxlen=c_pad, dtype=float)
            targets.append(span_expanded)

            h_emb.append(history_inputs['history'])
            h_pos.append(history_inputs['history_pos'])
            h_ent.append(history_inputs['history_ent'])
            cids.append(cid)
            tids.append(tid)
            context_map[cid] = context_inputs

        c_emb, c_pos, c_ent = [], [], []
        for cid in cids:
            c_emb.append(context_map[cid]['context'])
            c_pos.append(context_map[cid]['context_pos'])
            c_ent.append(context_map[cid]['context_ent'])

        h_emb, h_pos, h_ent = np.array(h_emb), np.array(h_pos), np.array(h_ent)
        c_emb, c_pos, c_ent = np.array(c_emb), np.array(c_pos), np.array(c_ent)
        h_emb, c_emb = np.concatenate((h_emb, h_pos, h_ent), axis=2), np.concatenate((c_emb, c_pos, c_ent), axis=2)
        targets = np.array(targets)
        return (cids, tids, h_emb, c_emb, targets)

    def generate_history_sequence(self, prev, current, questions, responses, h_pad):
        history, history_pos, history_ent = [], [], []
        context_emb, questions_emb, responses_emb = self.context_emb, self.questions_emb, self.responses_emb
        for p in prev:
            if p == None:
                continue
            else:
                history.append(self.questions_emb[p])
                history.append(np.zeros(shape=(1,1024)))
                history.append(self.responses_emb[p])
                history.append(np.zeros(shape=(1,1024)))
                history_pos.append(questions[p]['pos'])
                history_pos.append(['END'])
                history_pos.append(responses[p]['pos'])
                history_pos.append(['END'])
                history_ent.append(questions[p]['ent'])
                history_ent.append(['END'])
                history_ent.append(responses[p]['ent'])
                history_ent.append(['END'])

        # current
        history.append(self.questions_emb[current])
        history_pos.append(questions[current]['pos'])
        history_ent.append(questions[current]['ent'])
        history = np.concatenate(history, axis=0)
        history_pos = np.concatenate(history_pos, axis=0)
        history_ent = np.concatenate(history_ent, axis=0)
        span = np.array(responses[current]['answer_span'])
        assert history.shape[0] == history_pos.shape[0] == history_ent.shape[0]

        # history padding
        history = pad_sequences(history.T, maxlen=h_pad, dtype=float).T
        history_pos = list(pad_sequences(np.expand_dims(history_pos, axis=0), maxlen=h_pad, dtype=object, value='PAD')[0])
        history_ent = list(pad_sequences(np.expand_dims(history_ent, axis=0), maxlen=h_pad, dtype=object, value='PAD')[0])

        # one hot
        for x in range(len(history_pos)):
            p, e = history_pos[x], history_ent[x]
            history_pos[x] = self.pos_map[p]
            history_ent[x] = self.ent_map[e]

        return (history, history_pos, history_ent, span)

    def extract_sentences(self, sentences, words):
        sents = []
        for start, end in sentences:
            sents.append(words[start:end])
        return sents

    def start_pipeline(self, limit):
        return self.prepare_training(limit=limit)