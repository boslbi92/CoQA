from model.model_bidaf import BiDAF
from tqdm import tqdm
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os, json

def prepare_toy_data():
    path = '/Users/jason/Documents/Research/Thesis/CoQA/HenNet/data/coqa-dev-preprocessed.json'
    with open(path) as f:
        data = json.load(f)

    # load word2vec
    w2v = KeyedVectors.load_word2vec_format(fname='/Users/jason/Documents/Research/Dataset/Word2Vec.bin', binary=True, limit=50000)
    for i, x in enumerate(data['data']):
        context, questions, answers = None, [], []

        if i == 5: break

        for k, v in x['annotated_context'].items():
            if k == 'word':
                context = v
        for history in x['qas']:
            q, a = history['annotated_question'], history['annotated_answer']
            questions.append(q['word'])
            answers.append(a['word'])

        # convert to vectors
        if len(context) >= 500:
            context = context[0:500]
        else:
            context = context + (['_PAD_'] * (500 - len(context)))
        for i in range(len(context)):
            try:
                context[i] = w2v[context[i]]
            except:
                context[i] = np.zeros(300)
                continue

        for index in range(len(questions)):
            q = questions[index]
            if len(q) >= 20:
                q = q[0:20]
            else:
                q = q + (['_PAD_']*(20-len(q)))
            for i in range(len(q)):
                try:
                    q[i] = w2v[q[i]]
                except:
                    q[i] = np.zeros(300)
                    continue
            questions[index] = q


        # for index in range(len(answers)):
        #     a = answers[index]
        #     if len(a) >= 20:
        #         a = a[0:20]
        #     else:
        #         a = a + (['_PAD_']*(20-len(a)))
        #     for i in range(len(a)):
        #         try:
        #             a[i] = w2v[a[i]]
        #         except:
        #             a[i] = np.zeros(300)
        #             continue
        #     answers[index] = a

        train_context, train_question, test = [], [], []
        context, questions, answers = np.array(context), np.array(questions), np.array(answers)
        for q, a in zip(questions, answers):
            train_context.append(context)
            train_question.append(q)
            test.append(' '.join(a))

    return (np.array(train_context), np.array(train_question), np.array(test))

def main():
    context_in, history_in, test = prepare_toy_data()

    context_dim = context_in.shape
    rand_test = np.random.rand(context_dim[0], context_dim[1], 2)
    span_start, span_end = rand_test[:, :, 0], rand_test[:, :, 1]

    bidaf = BiDAF()

    # test span
    span_begin_probs = np.array([0.1, 0.3, 0.05, 0.3, 0.25])
    span_end_probs = np.array([0.5, 0.1, 0.2, 0.05, 0.15])
    best_span = bidaf.get_best_span(span_start[0], span_end[0])
    print (best_span)

    bidaf.build_model(context_input=context_in, history_input=history_in, span_start_out=span_start, span_end_out=span_end, epochs=1)
    return

main()