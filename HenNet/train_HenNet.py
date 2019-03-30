from model.model_bidaf import BiDAF
from tqdm import tqdm
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import os, json, argparse

def prepare_toy_data(num_conv=5, w2v_path='', w2v_limit=50000):
    path = os.getcwd() + '/data/coqa-dev-preprocessed.json'
    with open(path) as f:
        data = json.load(f)

    # load word2vec
    w2v_path = '/Users/jason/Documents/Research/Dataset/Word2Vec.bin'
    w2v = KeyedVectors.load_word2vec_format(fname=w2v_path, binary=True, limit=w2v_limit)

    train_context, train_question, train_begin, train_end = [], [], [], []
    max_context = 0
    for i, x in enumerate(data['data']):
        context, questions, answers, answers_span = None, [], [], []
        if i == num_conv: break

        for k, v in x['annotated_context'].items():
            if k == 'word':
                context = v
        for history in x['qas']:
            q, a = history['annotated_question'], history['annotated_answer']
            questions.append(q['word'])
            answers.append(a['word'])
            answers_span.append(history['answer_span'])

        if len(context) >= max_context:
            max_context = len(context)

        # convert to vectors
        if len(context) >= 1200:
            context = context[0:1200]
        else:
            context = context + (['_PAD_'] * (1200 - len(context)))
        for i in range(len(context)):
            try:
                context[i] = w2v[context[i]]
            except:
                context[i] = np.zeros(300)
                continue

        for index in range(len(questions)):
            q = questions[index]
            if len(q) >= 30:
                q = q[0:30]
            else:
                q = q + (['_PAD_']*(30-len(q)))
            for i in range(len(q)):
                try:
                    q[i] = w2v[q[i]]
                except:
                    q[i] = np.zeros(300)
                    continue
            questions[index] = q

        context, questions, answers_text, answers_span = np.array(context), np.array(questions), np.array(answers), np.array(answers_span)
        for q, a in zip(questions, answers_span):
            train_context.append(context)
            train_question.append(q)
            true_begin, true_end = a[0], a[1]
            initial_begin, initial_end = np.zeros(shape=(1200,)), np.zeros(shape=(1200,))
            initial_begin[true_begin] = 1.0
            initial_end[true_end] = 1.0
            train_begin.append(initial_begin)
            train_end.append(initial_end)
    return (np.array(train_context), np.array(train_question), np.array(train_begin), np.array(train_end))

def main():
    # argparser
    parser = argparse.ArgumentParser(description='HenNet Trainer')
    embedding_path = os.getcwd() + '/embeddings/Word2Vec.bin'
    parser.add_argument("-n", help='number of convs to train', type=int, default=9999999)
    parser.add_argument("-p", help='path of Word2Vec bin', type=str, default=embedding_path)
    parser.add_argument("-l", help='w2v limit', type=int, default=9999999)
    args = parser.parse_args()

    # trainer
    context_in, history_in, train_begin, train_end = prepare_toy_data(num_conv=args.n, w2v_path=args.p, w2v_limit=args.l)
    probs = np.stack([train_begin, train_end], axis=1)

    bidaf = BiDAF()
    bidaf.build_model(context_input=context_in, history_input=history_in, output=probs, epochs=50)
    return

main()