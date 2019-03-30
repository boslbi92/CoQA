import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
import re, sys, time

try:
    from HenNet.model_HenNet.han import HAN, AttentionLayer
    from HenNet.model_HenNet.henNet import HenNet, AttentionLayer
    from HenNet.utils import build_word2vec
except:
    from han import HAN, AttentionLayer
    from henNet import HenNet, AttentionLayer
    from utils import build_word2vec

tensorboard = TensorBoard(log_dir="train_logs/{}".format(time.time()))

def process_toy_data():
    max_words_per_sent, max_sent = 50, 15
    w2v_path = '/Users/jason/Documents/Research/Dataset/Word2Vec.bin'
    data_path = '/Users/jason/Documents/Research/Thesis/CoQA/HenNet/data/labeledTrainData.tsv'
    movie_data = pd.read_csv(data_path, sep='\t', nrows=50)

    # preprocess
    raw_text = movie_data['review'].values
    for x in range(len(raw_text)):
        text = raw_text[x]
        text = re.sub(r"\\", "", text)
        text = re.sub(r"\'", "", text)
        text = re.sub(r"\"", "", text)
        text = re.compile(r'<.*?>').sub('', text)
        raw_text[x] = text.lower()
    movie_data['clean_review'] = raw_text
    target, reviews = movie_data['sentiment'].values, movie_data['clean_review'].values

    # tokenize and embeddings
    word_tokenizer = Tokenizer(num_words=20000)
    word_tokenizer.fit_on_texts(reviews)
    X = np.zeros((len(reviews), max_sent, max_words_per_sent), dtype='int32')

    print ('Processing data input ...')
    for i, review in enumerate(tqdm(reviews, total=len(reviews))):
        sentences = sent_tokenize(review)
        tokenized_sentences = word_tokenizer.texts_to_sequences(sentences)
        tokenized_sentences = pad_sequences(tokenized_sentences, maxlen=max_words_per_sent)
        pad_size = max_sent - tokenized_sentences.shape[0]

        if pad_size < 0:
            tokenized_sentences = tokenized_sentences[0:max_sent]
        else:
            tokenized_sentences = np.pad(tokenized_sentences, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
        X[i] = tokenized_sentences[None, ...]
    y = to_categorical(target)

    # prepare embeddings
    embedding_dict, embedding_matrix = build_word2vec(word_index=word_tokenizer.word_index, fname=w2v_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # train HAN
    han = HenNet(max_words=max_words_per_sent, max_sentences=max_sent, output_size=2, embedding_matrix=embedding_matrix,
              word_encoding_dim=100, sentence_encoding_dim=100)
    han.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    han.summary()
    # han.fit(X_train, y_train, batch_size=10, epochs=50, validation_data=(X_test, y_test), callbacks=[tensorboard])
    return


process_toy_data()


