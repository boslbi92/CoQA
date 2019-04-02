import numpy as np
from tqdm import tqdm
from bert_serving.client import BertClient
from bert_serving.server import BertServer
from collections import OrderedDict
import os, json, pickle, time

# bert-serving-start -model_dir /media/Ubuntu/Research/Thesis/data/BERT-cased-large/ -num_worker=4 -pooling_strategy=NONE -max_seq_len=NONE -show_tokens_to_client -pooling_layer -4 -3 -2 -1 -cpu
class CoQAPreprocessor():
    def __init__(self):
        self.context_len, self.query_len, self.history_len = 0, 0, 0
        self.lemma_dict = dict()
        self.context_dict = dict()
        self.embeddings = dict()
        self.lemma_id = 1
        self.data_path = os.getcwd() + '/data/'
        self.embedding_path = os.getcwd() + '/embeddings/'


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

        with open(self.embedding_path + 'bert_embeddings_dev.pickle', 'wb') as f:
            pickle.dump(embeddings, f, protocol=3)

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

    def save_data(self, train, test, option, save):
        if save:
            print ('saving data ...')
            data_path = self.data_path

            with open(data_path + 'processed-{}-train.json'.format(option), 'w') as outfile:
                json.dump(train, outfile, indent=4)
            with open(data_path + 'processed-{}-test.json'.format(option), 'w') as outfile:
                json.dump(test, outfile, indent=4)
            with open(data_path + 'lemma_dict.json'.format(option), 'w') as outfile:
                json.dump(self.lemma_dict, outfile, indent=4)
            with open(data_path + 'context_dict.json'.format(option), 'w') as outfile:
                json.dump(self.context_dict, outfile, indent=4)
        return

    def process_CoQA(self, option='dev', save=False):
        # read data
        path = os.getcwd() + '/data/coqa-{}-preprocessed.json'.format(option)
        with open(path) as f:
            print('Reading json file ...')
            j = json.load(f)
            num_conv = len(j['data'])
            print ('Read {} conversations\n'.format(num_conv))

        train, test = [], []

        # process each conversation
        time.sleep(0.5)
        for x in tqdm(j['data'], total=num_conv):
            # context preprocessing
            original_context, annotated_context, context_id = x['context'], x['annotated_context'], x['id']
            words = annotated_context['lemma']
            lemma_id = self.add_to_lemma_map(words)
            char_id, ent_id, pos_id = annotated_context['charid'], annotated_context['ent_id'], annotated_context['pos_id']
            context_info = {'context_words': words, 'context_char': char_id, 'context_entity': ent_id, 'context_pos': pos_id, 'raw': original_context, 'id': context_id, 'lemma_id': lemma_id}
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
                targets.append((span, a_info['raw']))

                if len(q_words) > self.query_len:
                    self.query_len = len(q_words)

            contextualized_samples = self.contextualize(context_info, queries, answers, window=3)
            train += contextualized_samples
            test += targets

        time.sleep(0.5)
        assert len(test) == len(train)
        print ('Total {} turns processed'.format(len(test)))
        print ('max context words : {}, max history words : {}, max query words : {}'.format(self.context_len+1, self.history_len+1, self.query_len+1))
        print('vocab size : {}'.format(len(self.lemma_dict)+1))

        self.save_data(train, test, option, save)
        return (train, test)

    def contextualize(self, context, queries, answers, window=1):
        # contextual appending
        contextualized_samples, targets = [], []

        for i in range(1, len(queries)+1):
            if i == 1:
                prev_queries, prev_answers = [], []
                current_query, current_answer = queries[:i][-window:][0], answers[:i][-window:][0]
            else:
                prev_queries, prev_answers = queries[:i][-window:], answers[:i][-window:]
                current_query, current_answer = prev_queries.pop(), prev_answers.pop()

            # prev turns
            history_info = {'h_words': [], 'h_char': [], 'h_entity': [], 'h_pos': [], 'raw': '', 'turn': []}
            for j in range(len(prev_queries)):
                q, a = prev_queries[j], prev_answers[j]
                history_info['h_words'] += q['q_words'] + ['<Q-END>'] + a['a_words'] + ['<A-END>']
                history_info['h_char'] += q['q_char'] + [[-1]] + a['a_char'] + [[-1]]
                history_info['h_entity'] += q['q_entity'] + [-1] + a['a_entity'] + [-1]
                history_info['h_pos'] += q['q_pos'] + [-1] + a['a_pos'] + [-1]
                history_info['raw'] += q['raw'] + ' <Q-END> ' + a['raw'] + ' <A-END> '
                history_info['turn'] += [q['turn']] + ['<Q-END>'] + [a['turn']] + ['<A-END>']

            # current info
            history_info['context_id'] = context['id']
            history_info['h_words'] += current_query['q_words'] + ['<Q-END>']
            history_info['h_char'] += current_query['q_char'] + [[-1]]
            history_info['h_entity'] += current_query['q_entity'] + [-1]
            history_info['h_pos'] += current_query['q_pos'] + [-1]
            history_info['raw'] += current_query['raw'] + ' <Q-END>'
            history_info['turn'] += [current_query['turn']] + ['<Q-END>']

            if len(history_info['h_pos']) > self.history_len:
                self.history_len = len(history_info['h_pos'])

            lemma_id = self.add_to_lemma_map(history_info['h_words'])
            history_info['lemma_id'] = lemma_id
            contextualized_samples.append(history_info)
        return (contextualized_samples)

# preprocessor = CoQAPreprocessor()
# train, test = preprocessor.process_CoQA(save=True)
# preprocessor.build_BERT_embeddings()