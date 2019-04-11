from model.model_HenNet import HenNet
from tqdm import tqdm
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from preprocess import CoQAPreprocessor
import numpy as np
import os, json, argparse, pickle, time

def main():
    # argparser
    parser = argparse.ArgumentParser(description='HenNet Trainer')
    parser.add_argument("-c", help='number of convs to train', type=int, default=300)
    parser.add_argument("-b", help='batch size', type=int, default=25)
    args = parser.parse_args()

    prep = CoQAPreprocessor()
    cids, c_emb, c_pos, c_ent, h_emb, h_pos, h_ent, targets = prep.start_pipeline(limit=args.c)

    print ('context embedding : {}'.format(c_emb.shape))
    print ('context pos : {}'.format(c_pos.shape))
    print ('context ent : {}'.format(c_ent.shape))
    print ('history embedding : {}'.format(h_emb.shape))
    print ('history pos : {}'.format(h_pos.shape))
    print ('history ent : {}'.format(h_ent.shape))
    print ('span : {}'.format(targets.shape))

    time.sleep(2.0)
    print('Training HenNet ...\n')
    hn = HenNet()
    hn.build_model(context_input=c_emb, history_input=h_emb, output=targets, batch=args.b, epochs=100, shuffle=True)
    return

main()