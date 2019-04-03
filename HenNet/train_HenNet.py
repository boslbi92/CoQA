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
    ce, cnlp, he, hnlp, s = prep.start_pipeline()

    hn = HenNet()
    hn.build_model(context_input=ce, history_input=he, output=s, epochs=100)
    return

main()