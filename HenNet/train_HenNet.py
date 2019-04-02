from model.model_HenNet import HenNet
from tqdm import tqdm
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from preprocess import CoQAPreprocessor
import numpy as np
import os, json, argparse, pickle

def prepare_coqa_data():
    data_path = os.getcwd() + '/data/training/'
    with open(data_path + 'train-context-emb.pickle', 'rb') as f:
        c_words = pickle.load(f)
    with open(data_path + 'train-context-nlp.pickle', 'rb') as f:
        c_nlp = pickle.load(f)
    with open(data_path + 'train-history-emb.pickle', 'rb') as f:
        h_words = pickle.load(f)
    with open(data_path + 'train-history-nlp.pickle', 'rb') as f:
        h_nlp = pickle.load(f)
    with open(data_path + 'train-spans.pickle', 'rb') as f:
        spans = pickle.load(f)
    with open(data_path + 'bert-emb.pickle', 'rb') as f:
        emb_matrix = pickle.load(f)
    return (c_words, c_nlp, h_words, h_nlp, spans, emb_matrix)

def main():
    # trainer
    c_words, c_nlp, h_words, h_nlp, spans, emb_matrix = prepare_coqa_data()

    hn = HenNet()
    hn.build_model(context_input=context_in, history_input=history_in, output=probs, epochs=50)
    return

main()