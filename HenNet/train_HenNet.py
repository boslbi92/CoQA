import numpy as np
from model.model_HenNet import HenNet
from keras.preprocessing.sequence import pad_sequences
from preprocess import CoQAPreprocessor
from generator import CoQAGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
from model.metrics.custom_metrics import monitor_span
import os, json, argparse, pickle, time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tensorboard = TensorBoard(log_dir='train_logs/{}'.format(time.time()))
model_dir = os.getcwd() + '/saved_models/HenNet-{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(model_dir, monitor='val_loss', save_best_only=True)

def main():
    # argparser
    parser = argparse.ArgumentParser(description='HenNet Trainer')
    parser.add_argument("-b", help='batch size', type=int, default=100)
    parser.add_argument("-c", help='context pad size', type=int, default=600)
    parser.add_argument("-q", help='query pad size', type=int, default=100)
    parser.add_argument("-p", help='bert embedding directory', type=str, default=(os.getcwd()+'/data/bert/'))
    args = parser.parse_args()

    # generator and dev set
    train_generator = CoQAGenerator(option='train', batch=args.b, c_pad=args.c, h_pad=args.q, bert_path=args.p)
    dev = CoQAPreprocessor(option='dev', c_pad=args.c, h_pad=args.q, bert_path=args.p)
    val_cids, val_tids, val_c_emb, val_c_nlp, val_h_emb, val_h_nlp, val_targets = dev.start_pipeline(limit=1000000)

    # write ids
    print ('writing ids')
    with open(os.getcwd() + '/train_logs/val_ids.txt', 'w') as f:
        for c, d in zip(val_cids, val_tids):
            l = str(c) + '\t' + str(d) + '\n'
            f.write(l)
    f.close()

    print ('-'*100)
    print ('dev dimension')
    print ('context bert dim : {}'.format(val_c_emb.shape))
    print ('context nlp dim : {}'.format(val_c_nlp.shape))
    print ('history bert dim : {}'.format(val_h_emb.shape))
    print ('history nlp dim : {}'.format(val_h_nlp.shape))
    print ('span dim : {}'.format(val_targets.shape))
    print ('val turns : {}'.format(len(val_tids)))
    print ('-'*100)

    print('Training HenNet ...\n')
    time.sleep(1.0)
    hn = HenNet(c_pad=args.c, h_pad=args.q, nlp_dim=val_c_nlp.shape[-1])
    H = hn.build_model()
    H.fit_generator(train_generator, validation_data=([val_h_emb, val_c_emb, val_h_nlp, val_c_nlp], [val_targets]), epochs=50, steps_per_epoch=len(train_generator),
                    shuffle=True, use_multiprocessing=True, workers=4, callbacks=[monitor_span(), checkpoint])
    return


main()