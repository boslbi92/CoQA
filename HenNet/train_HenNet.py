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
    parser.add_argument("-c", help='number of convs to train', type=int, default=1000000)
    parser.add_argument("-b", help='batch size', type=int, default=100)
    args = parser.parse_args()

    prep = CoQAPreprocessor(option='train')
    train_cids, train_c_emb, train_c_pos, train_c_ent, train_h_emb, train_h_pos, train_h_ent, train_targets = prep.start_pipeline(limit=args.c)

    print ('-'*100)
    print ('training data information')
    print ('context embedding : {}'.format(train_c_emb.shape))
    print ('context pos : {}'.format(train_c_pos.shape))
    print ('context ent : {}'.format(train_c_ent.shape))
    print ('history embedding : {}'.format(train_h_emb.shape))
    print ('history pos : {}'.format(train_h_pos.shape))
    print ('history ent : {}'.format(train_h_ent.shape))
    print ('span : {}'.format(train_targets.shape))
    print ('-'*100)

    time.sleep(2.0)

    prep = CoQAPreprocessor(option='dev')
    dev_cids, dev_c_emb, dev_c_pos, dev_c_ent, dev_h_emb, dev_h_pos, dev_h_ent, dev_targets = prep.start_pipeline(limit=args.c)

    print ('-'*100)
    print ('dev data information')
    print ('context embedding : {}'.format(dev_c_emb.shape))
    print ('context pos : {}'.format(dev_c_pos.shape))
    print ('context ent : {}'.format(dev_c_ent.shape))
    print ('history embedding : {}'.format(dev_h_emb.shape))
    print ('history pos : {}'.format(dev_h_pos.shape))
    print ('history ent : {}'.format(dev_h_ent.shape))
    print ('span : {}'.format(dev_targets.shape))
    print ('-'*100)

    time.sleep(2.0)

    print('Training HenNet ...\n')
    hn = HenNet()
    hn.build_model(train_context=train_c_emb, train_history=train_h_emb, train_span=train_targets,
                   dev_context=dev_c_emb, dev_history=dev_h_emb, dev_span=dev_targets,
                   batch=args.b, epochs=100)
    return

main()