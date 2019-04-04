# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm
from bert_serving.client import BertClient
from bert_serving.server import BertServer
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
import os, json, pickle, time

# bert-serving-start -model_dir /media/Ubuntu/Research/Thesis/data/BERT-cased-large/ -num_worker=4 -pooling_strategy=NONE -max_seq_len=NONE
# -show_tokens_to_client -pooling_layer -4 -3 -2 -1 -cpu
class CoQAPreprocessor():
    def __init__(self):
        self.context_len, self.query_len, self.history_len = 0, 0, 0
        self.context_data, self.embeddings = {}, {}
        self.pos_labels, self.ner_labels = Counter(), Counter()
        self.data_path = os.getcwd() + '/data/processed/'
        self.embedding_path = os.getcwd() + '/embeddings/'
        self.train_path = os.getcwd() + '/data/training/'
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

    def add_to_labels(self, labels, type=None):
        if type == 'pos':
            label_map = self.pos_labels
        if type == 'ner':
            label_map = self.ner_labels
        for l in labels:
            label_map[l] += 1
        return

    def load_training_data(self):
        print('loading training data ...\n')
        data_path = self.train_path
        with open(data_path + 'train-context-emb.pickle', 'rb') as f:
            ce = pickle.load(f)
        with open(data_path + 'train-context-nlp.pickle', 'rb') as f:
            cnlp = pickle.load(f)
        with open(data_path + 'train-history-emb.pickle', 'rb') as f:
            he = pickle.load(f)
        with open(data_path + 'train-history-nlp.pickle', 'rb') as f:
            hnlp = pickle.load(f)
        with open(data_path + 'train-spans.pickle', 'rb') as f:
            s = pickle.load(f)

        print ('context embedding dim: {}'.format(ce.shape))
        print ('c-nlp embedding dim: {}'.format(cnlp.shape))
        print ('history embedding dim: {}'.format(he.shape))
        print ('h-nlp embedding dim: {}'.format(hnlp.shape))
        print ('span embedding dim: {}\n'.format(s.shape))
        return (ce, cnlp, he, hnlp, s)

    def save_processed_data(self, train_context, train_history, spans, stats, option, save):
        if save:
            print ('saving data ...')
            data_path = self.data_path
            with open(data_path + '{}-processed-context.json'.format(option), 'w') as f:
                json.dump(train_context, f, indent=4)
            with open(data_path + '{}-stats.json'.format(option), 'w') as f:
                json.dump(stats, f, indent=4)
            with open(data_path + '{}-processed-history.pickle'.format(option), 'wb') as f:
                pickle.dump(train_history, f, protocol=4)
            with open(data_path + '{}-context-map.pickle'.format(option), 'wb') as f:
                pickle.dump(self.context_data, f, protocol=4)
            with open(data_path + '{}-processed-spans.pickle'.format(option), 'wb') as f:
                pickle.dump(spans, f, protocol=4)
            print ('saving completed\n')
        return

    def process_CoQA(self, option='dev', save=False, conv_limit=999999):
        # read data
        path = os.getcwd() + '/data/coqa-{}-preprocessed.json'.format(option)
        with open(path) as f:
            print('Reading json file ...')
            j = json.load(f)
            num_conv = len(j['data'])
            print ('Read {} conversations\n'.format(num_conv))

        train_context, train_history, test = [], [], []
        count = 0

        print ('preparing BERT embedding ...')
        # process each conversation
        time.sleep(0.5)
        for x in tqdm(j['data'], total=num_conv):
            if count == conv_limit:
                break

            # context preprocessing
            original_context, annotated_context, context_id = x['context'], x['annotated_context'], x['id']
            words, sents, char_id, ent_id, pos_id = annotated_context['lemma'], annotated_context['sentences'], annotated_context['charid'], annotated_context['ent_id'], annotated_context['pos_id']
            sents = self.extract_sentences(sents, words)
            context_embedding = self.get_contextual_embeddings(sents, words)
            context_info = {'context_words': words, 'context_entity': ent_id, 'context_pos': pos_id, 'raw': original_context, 'id': context_id,
                            'num_words': len(words),'context_embedding': context_embedding}

            # compute max words in context
            if len(words) > self.context_len:
                self.context_len = len(words)

            # save context info in dict
            self.context_data[context_id] = context_info
            self.add_to_labels(context_info['context_pos'], type='pos')
            self.add_to_labels(context_info['context_entity'], type='ner')

            # history parsing for each turn
            queries, answers, targets = [], [], []
            for history in x['qas']:
                span = history['answer_span']
                q, a = history['annotated_question'], history['annotated_answer']
                raw_question, raw_answer = history['question'], history['raw_answer']
                q_words, q_char_id, q_ent_id, q_pos_id = q['lemma'], q['charid'], q['ent_id'], q['pos_id']
                a_words, a_char_id, a_ent_id, a_pos_id = a['word'], a['charid'], a['ent_id'], a['pos_id']

                q_info = {'q_words': q_words, 'q_char': q_char_id, 'q_entity': q_ent_id, 'q_pos': q_pos_id, 'raw': raw_question, 'turn': history['turn_id']}
                a_info = {'a_words': a_words, 'a_char': a_char_id, 'a_entity': a_ent_id, 'a_pos': a_pos_id, 'raw': raw_answer, 'turn': history['turn_id']}

                queries.append(q_info)
                answers.append(a_info)
                full_span = self.expand_spans(span, len(words), words, raw_answer)
                targets.append(full_span)

                # copmute max words in query
                if len(q_words) > self.query_len:
                    self.query_len = len(q_words)
            repeated_contexts, contextualized_samples = self.contextualize(context_id, queries, answers, window=3)

            # add contextualized samples
            train_context += repeated_contexts
            train_history += contextualized_samples
            test += targets
            count += 1

        time.sleep(0.5)
        assert len(test) == len(train_context) == len(train_history)
        print ('Total {} turns processed'.format(len(test)))
        print ('max context words : {}, max history words : {}, max query words : {}'.format(self.context_len+1, self.history_len+1, self.query_len+1))

        stats = {'pos_ids': len(self.pos_labels), 'ner_ids': len(self.ner_labels)}
        self.save_processed_data(train_context, train_history, test, stats, option, save)
        return

    def contextualize(self, context_id, queries, answers, window=3):
        # contextual appending
        repeat_contexts, contextualized_samples, targets = [], [], []

        for i in range(1, len(queries)+1):
            if i == 1:
                prev_queries, prev_answers = [], []
                current_query, current_answer = queries[:i][-window:][0], answers[:i][-window:][0]
            else:
                prev_queries, prev_answers = queries[:i][-window:], answers[:i][-window:]
                current_query, current_answer = prev_queries.pop(), prev_answers.pop()

            # prev turns
            history_info = {'h_words': [], 'h_entity': [], 'h_pos': [], 'raw': '', 'turn': []}
            for j in range(len(prev_queries)):
                q, a = prev_queries[j], prev_answers[j]
                history_info['h_words'] += q['q_words'] + ['<Q-END>'] + a['a_words'] + ['<A-END>']
                history_info['h_entity'] += q['q_entity'] + [-1] + a['a_entity'] + [-1]
                history_info['h_pos'] += q['q_pos'] + [-1] + a['a_pos'] + [-1]
                history_info['raw'] += q['raw'] + ' <Q-END> ' + a['raw'] + ' <A-END> '
                history_info['turn'] += [q['turn']] + ['<Q-END>'] + [a['turn']] + ['<A-END>']
                # history_info['h_char'] += q['q_char'] + [[-1]] + a['a_char'] + [[-1]]

            # current info
            history_info['context_id'] = context_id
            history_info['h_words'] += current_query['q_words'] + ['<Q-END>']
            history_info['h_entity'] += current_query['q_entity'] + [-1]
            history_info['h_pos'] += current_query['q_pos'] + [-1]
            history_info['raw'] += current_query['raw'] + ' <Q-END>'
            history_info['turn'] += [current_query['turn']] + ['<Q-END>']
            # history_info['h_char'] += current_query['q_char'] + [[-1]]
            history_info['history_embedding'] = self.get_contextual_embeddings([history_info['h_words']], history_info['h_words'])

            # compute max history words
            if len(history_info['h_words']) > self.history_len:
                self.history_len = len(history_info['h_words'])

            self.add_to_labels(history_info['h_pos'], type='pos')
            self.add_to_labels(history_info['h_entity'], type='ner')
            repeat_contexts.append(context_id)
            contextualized_samples.append(history_info)

        assert len(repeat_contexts) == len(contextualized_samples)
        return (repeat_contexts, contextualized_samples)

    def prepare_training_set(self, history_pad=75, context_pad=1010, save=False):
        print ('preparing training set ...')
        with open(self.data_path + 'dev-processed-context.json') as f:
            context_ids = json.load(f)
        with open(self.data_path + 'dev-processed-history.pickle', 'rb') as f:
            history = pickle.load(f)
            history = pd.DataFrame(history)
        with open(self.data_path + 'dev-context-map.pickle', 'rb') as f:
            context = pickle.load(f)
        with open(self.data_path + 'dev-processed-spans.pickle', 'rb') as f:
            test = np.array(pickle.load(f))

        repeat_context = []
        for cid in context_ids:
            repeat_context.append(context[cid])
        context = pd.DataFrame(repeat_context)

        # context padding
        print ('Preparing for context input ...')
        context_entity = pad_sequences(context['context_entity'].values, maxlen=context_pad, dtype=int, value=0)
        context_pos = pad_sequences(context['context_pos'].values, maxlen=context_pad, dtype=int, value=0)
        context_embedding = pad_sequences(context['context_embedding'].values.T, maxlen=context_pad, dtype=float, value=0.0)
        context_words = pad_sequences(context['context_words'].values, maxlen=context_pad, dtype=object, value=['[UNK]'])

        # context_embedding = np.apply_along_axis(self.build_bert, axis=1, arr=context_lemma_id, e=embeddings, lm=lemma_map, bert=bert_matrix)
        context_entity = np.apply_along_axis(self.slice, axis=1, arr=context_entity)
        context_pos = np.apply_along_axis(self.slice, axis=1, arr=context_pos)
        c_nlp_input = np.concatenate((context_entity, context_pos), axis=2)

        # history padding
        print ('Preparing for history input ...')
        h_entity = pad_sequences(history['h_entity'].values, maxlen=history_pad, dtype=int, value=0)
        h_pos = pad_sequences(history['h_pos'].values, maxlen=history_pad, dtype=int, value=0)
        h_embedding = pad_sequences(history['history_embedding'].values.T, maxlen=history_pad, dtype=float, value=0.0)

        # history_word_input = np.apply_along_axis(self.build_bert, axis=1, arr=lemma_id, e=embeddings, lm=lemma_map, bert=bert_matrix)
        h_entity = np.apply_along_axis(self.slice, axis=1, arr=h_entity)
        h_pos = np.apply_along_axis(self.slice, axis=1, arr=h_pos)
        h_nlp_input = np.concatenate((h_entity, h_pos), axis=2)

        # span padding
        print ('Preparing for span input ...')
        spans = pad_sequences(test.T, maxlen=context_pad, dtype=float, value=0.0)
        reshaped_spans = []
        with open(self.train_path + "spans.txt", 'w') as san:
            for i in range(spans.shape[0]):
                start, end = spans[i][:,0], spans[i][:,1]
                start_index, end_idnex = np.argmax(start), np.argmax(end)
                target = ' '.join(context_words[i][start_index:end_idnex+1])
                target = bytes(target, 'utf-8').decode('utf-8', 'ignore')
                san.write((target + '\n'))
                reshaped_spans.append([start, end])
            spans = np.array(reshaped_spans)
            san.close()

        time.sleep(1.0)
        if save:
            print('saving training data ...')
            data_path = self.train_path
            with open(data_path + 'train-context-emb.pickle', 'wb') as f:
                pickle.dump(context_embedding, f, protocol=4)
            with open(data_path + 'train-context-nlp.pickle', 'wb') as f:
                pickle.dump(c_nlp_input, f, protocol=4)
            with open(data_path + 'train-history-emb.pickle', 'wb') as f:
                pickle.dump(h_embedding, f, protocol=4)
            with open(data_path + 'train-history-nlp.pickle', 'wb') as f:
                pickle.dump(h_nlp_input, f, protocol=4)
            with open(data_path + 'train-spans.pickle', 'wb') as f:
                pickle.dump(spans, f, protocol=4)
            time.sleep(1.0)
            print('saving completed')

        return

    def extract_sentences(self, sentences, words):
        sents = []
        for start, end in sentences:
            sents.append(words[start:end])
        return sents

    def expand_spans(self, span, context_length, words, raw_answer):
        spans = np.zeros(shape=(context_length, 2))
        true_start, true_end = span[0], span[1]
        spans[true_start, 0] = 1.0
        spans[true_end, 1] = 1.0
        return spans

    def slice(self, X):
        expanded_dim = np.zeros(shape=(len(X), 1))
        for i in range(len(X)):
            expanded_dim[i, :] = X[i]
        return expanded_dim

    def start_pipeline(self, conv_limit=5, generate_data=False):
        if generate_data:
            self.process_CoQA(save=True, conv_limit=conv_limit)
            time.sleep(1.0)

        self.prepare_training_set(save=True)
        time.sleep(1.0)
        return self.load_training_data()