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
    parser.add_argument("-b", help='batch size', type=int, default=10)
    parser.add_argument("-g", help='generate data', type=bool, default=False)
    parser.add_argument("-tp", help='training path', type=str, required=True)
    args = parser.parse_args()

    prep = CoQAPreprocessor()
    ce, cnlp, he, hnlp, s = prep.start_pipeline(conv_limit=args.c, generate_data=args.g, training_path=args.tp)

    time.sleep(5.0)
    print('Training HenNet ...\n')
    hn = HenNet()
    hn.build_model(context_input=ce, history_input=he, output=s, epochs=100, batch=args.b, shuffle=False)
    return

main()