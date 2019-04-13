import numpy as np
import os, pickle
from tqdm import tqdm

def clean_bert(option):
    option, path = option, os.getcwd()+'/data/bert/'
    with open(path + 'context_{}.pickle'.format(option), 'rb') as f:
        context_emb = pickle.load(f)
    with open(path + 'questions_{}.pickle'.format(option), 'rb') as f:
        questions_emb = pickle.load(f)
    with open(path + 'responses_{}.pickle'.format(option), 'rb') as f:
        responses_emb = pickle.load(f)

    for k, v in tqdm(context_emb.items()):
        for x in range(v.shape[0]):
            word_v = v[x]
            sanity = np.isnan(word_v).any()
            if sanity == True:
                print (k)
                v[x] = np.zeros(1024)

    for k, v in tqdm(questions_emb.items()):
        for x in range(v.shape[0]):
            word_v = v[x]
            sanity = np.isnan(word_v).any()
            if sanity == True:
                print (k)
                v[x] = np.zeros(1024)

    for k, v in tqdm(responses_emb.items()):
        for x in range(v.shape[0]):
            word_v = v[x]
            sanity = np.isnan(word_v).any()
            if sanity == True:
                print (k)
                v[x] = np.zeros(1024)

    with open(path + 'context_{}.pickle'.format(option), 'wb') as f:
        pickle.dump(context_emb, f, protocol=4)
    with open(path + 'context_{}.pickle'.format(option), 'wb') as f:
        pickle.dump(context_emb, f, protocol=4)
    with open(path + 'context_{}.pickle'.format(option), 'wb') as f:
        pickle.dump(context_emb, f, protocol=4)

    print ('cleanup done!')
    return

clean_bert(option='train')