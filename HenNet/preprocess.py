import numpy as np
import pandas as pd
from tqdm import tqdm
from bert_serving.client import BertClient
from bert_serving.server import BertServer
from keras.preprocessing.sequence import pad_sequences
from collections import OrderedDict
import os, json, pickle, time

# bert-serving-start -model_dir /media/Ubuntu/Research/Thesis/data/BERT-cased-large/ -num_worker=4 -pooling_strategy=NONE -max_seq_len=NONE
# -show_tokens_to_client -pooling_layer -4 -3 -2 -1 -cpu
class CoQAPreprocessor():
    def __init__(self):
        self.context_len, self.query_len, self.history_len = 0, 0, 0
        self.lemma_dict = {'_PAD_': 0}
        self.context_dict = dict()
        self.embeddings = dict()
        self.lemma_id = 1
        self.data_path = os.getcwd() + '/data/processed/'
        self.embedding_path = os.getcwd() + '/embeddings/'
        self.train_path = os.getcwd() + '/data/training/'
        self.visited = {}

    def build_BERT_embeddings(self):
        with open(self.data_path + 'lemma_dict.json') as data_file:
            texts = json.load(data_file)

        texts = list(texts.keys())
        original_tokens = texts.copy()
        for x in range(len(texts)):
            texts[x] = [texts[x]]

        bc = BertClient()
        encoding = bc.encode(texts, is_tokenized=True, show_tokens=True)
        vectors, tokens = encoding[0], np.array(encoding[1])
        print('Embedding shape : {}'.format(vectors.shape))
        word_vectors, tokens = vectors[:, 1, :], tokens[:, 1]

        embeddings = dict()
        assert vectors.shape[1] == 3
        assert len(tokens) == word_vectors.shape[0] == len(original_tokens)
        for x in range(len(original_tokens)):
            ot, rt = original_tokens[x], tokens[x]
            if rt != '[UNK]':
                embeddings[ot] = word_vectors[x]
            else:
                embeddings[ot] = np.zeros(word_vectors.shape[1])

        print ('Embedding dict of size {} generated, writing to json ...'.format(len(embeddings)))
        with open(self.embedding_path + 'bert_embeddings_dev.pickle', 'wb') as f:
            pickle.dump(embeddings, f, protocol=4)

        return (embeddings)

    def add_to_lemma_map(self, words):
        lemma_id = []
        for w in words:
            if w not in self.lemma_dict:
                self.lemma_dict[w] = self.lemma_id
                lemma_id.append(self.lemma_id)
                self.lemma_id += 1
            else:
                lemma_id.append(self.lemma_dict[w])
        return lemma_id

    def save_processed_data(self, train_context, train_history, spans, option, save):
        if save:
            print ('saving data ...')
            data_path = self.data_path
            with open(data_path + '{}-processed-context.json'.format(option), 'w') as f:
                json.dump(train_context, f, indent=4)
            with open(data_path + '{}-processed-history.json'.format(option), 'w') as f:
                json.dump(train_history, f, indent=4)
            with open(data_path + '{}-processed-spans.pickle'.format(option), 'wb') as f:
                pickle.dump(spans, f, protocol=4)
            with open(data_path + 'lemma_dict.json'.format(option), 'w') as f:
                json.dump(self.lemma_dict, f, indent=4)
        return

    def process_CoQA(self, option='dev', save=False):
        # read data
        path = os.getcwd() + '/data/coqa-{}-preprocessed.json'.format(option)
        with open(path) as f:
            print('Reading json file ...')
            j = json.load(f)
            num_conv = len(j['data'])
            print ('Read {} conversations\n'.format(num_conv))

        train_context, train_history, test = [], [], []

        # process each conversation
        time.sleep(0.5)
        for x in tqdm(j['data'], total=num_conv):
            # context preprocessing
            original_context, annotated_context, context_id = x['context'], x['annotated_context'], x['id']
            words = annotated_context['lemma']
            lemma_id = self.add_to_lemma_map(words)
            char_id, ent_id, pos_id = annotated_context['charid'], annotated_context['ent_id'], annotated_context['pos_id']
            context_info = {'context_words': words, 'context_char': char_id, 'context_entity': ent_id,
                            'context_pos': pos_id, 'raw': original_context, 'id': context_id, 'lemma_id': lemma_id}
            self.context_dict[context_id] = context_info

            if len(words) > self.context_len:
                self.context_len = len(words)

            # history parsing for each turn
            queries, answers, targets = [], [], []
            for history in x['qas']:
                q, a = history['annotated_question'], history['annotated_answer']
                raw_question, raw_answer = history['question'], history['raw_answer']
                q_words, q_char_id, q_ent_id, q_pos_id = q['lemma'], q['charid'], q['ent_id'], q['pos_id']
                a_words, a_char_id, a_ent_id, a_pos_id = a['word'], a['charid'], a['ent_id'], a['pos_id']

                q_info = {'q_words': q_words, 'q_char': q_char_id, 'q_entity': q_ent_id, 'q_pos': q_pos_id, 'raw': raw_question, 'turn': history['turn_id']}
                a_info = {'a_words': a_words, 'a_char': a_char_id, 'a_entity': a_ent_id, 'a_pos': a_pos_id, 'raw': raw_answer, 'turn': history['turn_id']}
                span = history['answer_span']

                queries.append(q_info)
                answers.append(a_info)
                full_span = self.expand_spans(span, len(words), words, raw_answer)
                targets.append(full_span)

                if len(q_words) > self.query_len:
                    self.query_len = len(q_words)

            del context_info['context_char']
            repeated_contexts, contextualized_samples = self.contextualize(context_info, queries, answers, window=3)
            train_context += repeated_contexts
            train_history += contextualized_samples
            test += targets

        time.sleep(0.5)
        assert len(test) == len(train_context) == len(train_history)
        print ('Total {} turns processed'.format(len(test)))
        print ('max context words : {}, max history words : {}, max query words : {}'.format(self.context_len+1, self.history_len+1, self.query_len+1))
        print('vocab size : {}'.format(len(self.lemma_dict)+1))

        self.save_processed_data(train_context, train_history, test, option, save)
        return (train_context, train_history, test)

    def expand_spans(self, span, context_length, words, raw_answer):
        spans = np.zeros(shape=(context_length, 2))
        true_start, true_end = span[0], span[1]
        spans[true_start, 0] = 1.0
        spans[true_end, 1] = 1.0
        return spans

    def contextualize(self, context, queries, answers, window=1):
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
                # history_info['h_char'] += q['q_char'] + [[-1]] + a['a_char'] + [[-1]]
                history_info['h_entity'] += q['q_entity'] + [-1] + a['a_entity'] + [-1]
                history_info['h_pos'] += q['q_pos'] + [-1] + a['a_pos'] + [-1]
                history_info['raw'] += q['raw'] + ' <Q-END> ' + a['raw'] + ' <A-END> '
                history_info['turn'] += [q['turn']] + ['<Q-END>'] + [a['turn']] + ['<A-END>']

            # current info
            # history_info['context_id'] = context['id']
            history_info['h_words'] += current_query['q_words'] + ['<Q-END>']
            # history_info['h_char'] += current_query['q_char'] + [[-1]]
            history_info['h_entity'] += current_query['q_entity'] + [-1]
            history_info['h_pos'] += current_query['q_pos'] + [-1]
            history_info['raw'] += current_query['raw'] + ' <Q-END>'
            history_info['turn'] += [current_query['turn']] + ['<Q-END>']

            if len(history_info['h_pos']) > self.history_len:
                self.history_len = len(history_info['h_pos'])

            lemma_id = self.add_to_lemma_map(history_info['h_words'])
            history_info['lemma_id'] = lemma_id
            repeat_contexts.append(context)
            contextualized_samples.append(history_info)
        assert len(repeat_contexts) == len(contextualized_samples)
        return (repeat_contexts, contextualized_samples)

    def prepare_training_set(self, history_pad=75, context_pad=1010, save=False):
        with open(self.data_path + 'dev-processed-context.json') as data_file:
            context = pd.read_json(data_file)
        with open(self.data_path + 'dev-processed-history.json') as data_file:
            history = pd.read_json(data_file)
        with open(self.data_path + 'dev-processed-spans.pickle', 'rb') as data_file:
            test = np.array(pickle.load(data_file))
        with open(self.data_path + 'lemma_dict.json') as data_file:
            lemma_map = json.load(data_file)
            lemma_map = {v: k for k, v in lemma_map.items()}
        with open(self.embedding_path + 'bert_embeddings_dev.pickle', 'rb') as data_file:
            embeddings = pickle.load(data_file)

        # span padding
        print ('Preparing for span input ...')
        bert_matrix = np.zeros(shape=(len(embeddings)+1, 1024))
        spans = pad_sequences(test.T, maxlen=context_pad, dtype=float, value=0.0)

        # context padding
        print ('Preparing for context input ...')
        context_lemma_id = pad_sequences(context['lemma_id'].values, maxlen=context_pad, dtype=int, value=0)
        context_entity = pad_sequences(context['context_entity'].values, maxlen=context_pad, dtype=int, value=0)
        context_pos = pad_sequences(context['context_pos'].values, maxlen=context_pad, dtype=int, value=0)
        context_words = pad_sequences(context['context_words'].values, maxlen=context_pad, dtype=object, value='_PAD_')

        context_word_input = np.apply_along_axis(self.build_bert, axis=1, arr=context_lemma_id, e=embeddings, lm=lemma_map, bert=bert_matrix)
        context_entity = np.apply_along_axis(self.slice, axis=1, arr=context_entity)
        context_pos = np.apply_along_axis(self.slice, axis=1, arr=context_pos)
        c_nlp_input = np.concatenate((context_entity, context_pos), axis=2)

        # history padding
        print ('Preparing for history input ...')
        lemma_id = pad_sequences(history['lemma_id'].values, maxlen=history_pad, dtype=int, value=0)
        h_entity = pad_sequences(history['h_entity'].values, maxlen=history_pad, dtype=int, value=0)
        h_pos = pad_sequences(history['h_pos'].values, maxlen=history_pad, dtype=int, value=0)
        h_words = pad_sequences(history['h_words'].values, maxlen=history_pad, dtype=object, value='_PAD_')

        history_word_input = np.apply_along_axis(self.build_bert, axis=1, arr=lemma_id, e=embeddings, lm=lemma_map, bert=bert_matrix)
        h_entity = np.apply_along_axis(self.slice, axis=1, arr=h_entity)
        h_pos = np.apply_along_axis(self.slice, axis=1, arr=h_pos)
        h_nlp_input = np.concatenate((h_entity, h_pos), axis=2)

        if save:
            print('saving training data ...')
            data_path = self.train_path
            with open(data_path + 'train-context-emb.pickle', 'wb') as f:
                pickle.dump(context_word_input, f, protocol=4)
            with open(data_path + 'train-context-nlp.pickle', 'wb') as f:
                pickle.dump(c_nlp_input, f, protocol=4)
            with open(data_path + 'train-history-emb.pickle', 'wb') as f:
                pickle.dump(history_word_input, f, protocol=4)
            with open(data_path + 'train-history-nlp.pickle', 'wb') as f:
                pickle.dump(h_nlp_input, f, protocol=4)
            with open(data_path + 'train-spans.pickle', 'wb') as f:
                pickle.dump(spans, f, protocol=4)
            with open(data_path + 'bert-emb.pickle', 'wb') as f:
                pickle.dump(bert_matrix, f, protocol=4)
            print('saving completed')

        return (context_word_input, c_nlp_input, history_word_input, h_nlp_input, spans, bert_matrix)

    def slice(self, X):
        expanded_dim = np.zeros(shape=(len(X), 1))
        for i in range(len(X)):
            expanded_dim[i, :] = X[i]
        return expanded_dim

    def build_bert(self, lemma_id, e, lm, bert):
        for i in range(lemma_id.shape[0]):
            wid = lemma_id[i]
            if wid in self.visited:
                continue
            if wid == 0:
                continue
            else:
                v = e[lm[wid]]
                v_1, v_2, v_3, v_4 = v[0:1024], v[1024:2048], v[2048:3072], v[3072:4096]
                v = (v_1 + v_2 + v_3 + v_4)
                bert[wid] = v
            self.visited[wid] = 'key'
        return lemma_id

# preprocessor = CoQAPreprocessor()
# preprocessor.process_CoQA(save=True)
# context_word_input, c_nlp_input, history_word_input, h_nlp_input, spans, bert_matrix = preprocessor.prepare_training_set(save=False)