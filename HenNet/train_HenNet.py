from model.model_HenNet import HenNet
from tqdm import tqdm
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from preprocess import CoQAPreprocessor
import numpy as np
import os, json, argparse, pickle

def main():
    prep = CoQAPreprocessor()

    # trainer
    ce, cnlp, he, hnlp, s = prep.load_training_data()

    hn = HenNet()
    hn.build_model(context_input=ce, history_input=he, output=s, epochs=50)
    return

main()