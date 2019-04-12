# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence
from collections import Counter
import os, json, pickle, time, copy

class CoQAGenerator(Sequence):
    def __init__(self, option):
        self.option = option
        self.context_len, self.history_len = 0, 0
        self.c_pad, self.h_pad = 10, 10
        self.batch_size, self.num_samples = 50, 0
        self.exclude = []
        self.data_path = os.getcwd() + '/data/processed/'
        self.context_emb, self.questions_emb, self.responses_emb = self.load_embeddings()
        self.context, self.questions, self.responses = self.load_processed_data()
        self.train, self.test = self.join_by_id(self.context, self.questions, self.responses, window=3)

    def load_embeddings(self):
        path = os.getcwd() + '/data/bert/'
        option = self.option
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
            if num_words > 400:
                self.exclude.append(cid)

        for qid, v in q.items():
            cid = qid.split('_')[0]
            c[cid]['history'].append(qid)

        # exclude long sessions
        for cid in self.exclude:
            del c[cid]

        print ('{} conversations loaded'.format(len(c)))
        return (c, q, r)

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

    def __len__(self):
        n_batch = len(self.train) / self.batch_size
        return int(n_batch)

    def __getitem__(self, index, c_pad=500, h_pad=100, limit=500):
        context_emb, questions_emb, responses_emb = self.context_emb, self.questions_emb, self.responses_emb
        context, questions, responses = self.context, self.questions, self.responses
        train, test, batch = self.train, self.test, self.batch_size

        start_index = (index * self.batch_size)
        end_index = ((index + 1) * self.batch_size)
        self.indices = np.arange(len(train))
        inds = self.indices[start_index:end_index]

        iteration = min(limit, len(train))
        context_map, targets = {}, []
        h_emb, h_pos, h_ent, cids, tids = [], [], [], [], []
        c_emb, c_pos, c_ent = [], [], []

        if end_index >= len(train):
            end_index = len(train)

        for i in range(start_index, end_index):
            cid, history, tid = train[i][0], train[i][1], train[i][1][-1]
            history_inputs = {'context_id': cid, 'history': [], 'history_pos': [], 'history_ent': []}
            context_inputs = {'context': [], 'context_pos': [], 'context_ent': []}

            # context input
            context_inputs['context'] = context_emb[cid]
            context_inputs['context_pos'] = np.array(context[cid]['pos'])
            context_inputs['context_ent'] = np.array(context[cid]['ent'])

            # context padding
            context_inputs['context'] = pad_sequences(context_inputs['context'].T, maxlen=c_pad, dtype=float, value=0.0).T
            context_inputs['context_pos'] = pad_sequences(np.expand_dims(context_inputs['context_pos'], axis=0), maxlen=c_pad, dtype=object, value='PAD')[0]
            context_inputs['context_ent'] = pad_sequences(np.expand_dims(context_inputs['context_ent'], axis=0), maxlen=c_pad, dtype=object, value='PAD')[0]

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
            context_map[cid] = context_inputs
            cids.append(cid)
            tids.append(tid)

        # context repeat
        for cid in cids:
            c_emb.append(context_map[cid]['context'])
            c_pos.append(context_map[cid]['context_pos'])
            c_ent.append(context_map[cid]['context_ent'])
        h_emb, h_pos, h_ent = np.array(h_emb), np.array(h_pos), np.array(h_ent)
        c_emb, c_pos, c_ent = np.array(c_emb), np.array(c_pos), np.array(c_ent)
        targets = np.array(targets)
        # print (h_emb.shape, c_emb.shape, targets.shape)
        # print (inds)
        return [h_emb, c_emb], [targets]

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
        history_pos = pad_sequences(np.expand_dims(history_pos, axis=0), maxlen=h_pad, dtype=object, value='PAD')[0]
        history_ent = pad_sequences(np.expand_dims(history_ent, axis=0), maxlen=h_pad, dtype=object, value='PAD')[0]
        return (history, history_pos, history_ent, span)
