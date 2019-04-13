import numpy as np
import os, pickle, argparse, time
from tqdm import tqdm

def clean_bert(option, path):
    print ('cleanup start')

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
        print ('saving context done')
    time.sleep(1.0)
    with open(path + 'questions_{}.pickle'.format(option), 'wb') as f:
        pickle.dump(questions_emb, f, protocol=4)
        print ('saving questions done')
    time.sleep(1.0)
    with open(path + 'responses_{}.pickle'.format(option), 'wb') as f:
        pickle.dump(responses_emb, f, protocol=4)
        print ('saving responses done')
    time.sleep(1.0)

    print ('cleanup done!')
    return

def main():
    parser = argparse.ArgumentParser(description='Bert cleanup')
    parser.add_argument("-p", help='bert embedding directory', type=str, required=True)
    args = parser.parse_args()
    clean_bert(option='train', path=args.p)
    clean_bert(option='dev', path=args.p)

main()