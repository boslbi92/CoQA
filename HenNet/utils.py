import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models import KeyedVectors
from bert_serving.client import BertClient
from bert_serving.server import BertServer


def build_word2vec(word_index, fname):
    print('Reading word2vec ...')
    w2v = KeyedVectors.load_word2vec_format(fname, binary=True, limit=5000)

    embeddings_dict = dict()
    embedding_matrix = np.zeros((len(word_index)+1, 300))

    for word, i in word_index.items():
        if i >= len(word_index):
            continue
        try:
            v = w2v[word]
            embedding_matrix[i] = v
            embeddings_dict[word] = v
        except:
            continue
    return (embeddings_dict, embedding_matrix)

# bert-serving-start -model_dir /Users/jason/Documents/Research/Dataset/embeddings/BERT-cased-large/ -num_worker=4 -pooling_strategy NONE
def build_BERT_embeddings():
    bc = BertClient()


    texts = [['hey'], ['Hey'], ['I'], ['ate'], ['Mcdonalds'], ['mskeiwxcveftt'], [',']]
    vectors = bc.encode(texts, is_tokenized=True, show_tokens=True)
    return

def terminate_BERT_server(port):
    bs = BertServer()
    bs.shutdown(port=port)

build_BERT_embeddings()

# terminate_BERT_server(port=555)