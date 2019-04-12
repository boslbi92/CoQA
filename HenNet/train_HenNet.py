from model.model_HenNet import HenNet
from tqdm import tqdm
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from preprocess import CoQAPreprocessor
from generator import CoQAGenerator
import numpy as np
import os, json, argparse, pickle, time

def main():
    # argparser
    parser = argparse.ArgumentParser(description='HenNet Trainer')
    parser.add_argument("-c", help='number of convs to train', type=int, default=250)
    parser.add_argument("-b", help='batch size', type=int, default=100)
    args = parser.parse_args()

    train_generator = CoQAGenerator(option='dev')
    dev_generator = CoQAGenerator(option='dev')


    # write ids
    # print ('writing ids')
    # with open(os.getcwd() + '/train_logs/dev_ids.txt', 'w') as f:
    #     for c, d in zip(dev_cids, dev_tids):
    #         l = str(c) + '\t' + str(d) + '\n'
    #         f.write(l)
    # f.close()

    print('Training HenNet ...\n')
    hn = HenNet()
    H = hn.build_model()
    H.fit_generator(train_generator, validation_data=dev_generator, epochs=2, steps_per_epoch=len(train_generator),
                    shuffle=True, use_multiprocessing=True, workers=2)
    return

main()