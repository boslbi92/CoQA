import numpy as np
from model.model_HenNet import HenNet
from model.model_HenNet_GPU import HenNet_GPU
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
    parser.add_argument("-b", help='batch size', type=int, default=32)
    parser.add_argument("-c", help='context pad size', type=int, default=500)
    parser.add_argument("-q", help='query pad size', type=int, default=100)
    parser.add_argument("-p", help='bert embedding directory', type=str, default=(os.getcwd()+'/data/bert/'))
    parser.add_argument("-g", help='GPU mode', type=str, default='False', required=True)
    parser.add_argument("-d", help='hidden dimension scaling factor', type=int, default=4)
    args = parser.parse_args()

    # generator and dev set
    train_generator = CoQAGenerator(option='train', batch=args.b, c_pad=args.c, h_pad=args.q, bert_path=args.p)
    dev = CoQAPreprocessor(option='dev', c_pad=args.c, h_pad=args.q, bert_path=args.p)
    val_cids, val_tids, val_h_emb, val_c_emb, val_targets = dev.start_pipeline(limit=1000000)

    # write ids
    print ('writing ids')
    with open(os.getcwd() + '/train_logs/val_ids.txt', 'w') as f:
        for c, d in zip(val_cids, val_tids):
            l = str(c) + '\t' + str(d) + '\n'
            f.write(l)
    f.close()

    print ('-'*100)
    print ('dev dimension')
    print ('context dim : {}'.format(val_c_emb.shape))
    print ('history dim : {}'.format(val_h_emb.shape))
    print ('span dim : {}'.format(val_targets.shape))
    print ('val turns : {}'.format(len(val_tids)))
    print ('-'*100)

    if args.g.lower() == 'false':
        print('Training HenNet on CPU mode...\n')
        time.sleep(1.0)
        hn = HenNet(c_pad=val_c_emb.shape[-1], h_pad=args.q, hidden_scale=args.d)
        H = hn.build_model()
        H.fit_generator(train_generator, validation_data=([val_h_emb, val_c_emb], [val_targets]), epochs=50, steps_per_epoch=len(train_generator),
                        shuffle=True, use_multiprocessing=True, workers=6, callbacks=[monitor_span(), checkpoint])
    elif args.g.lower() == 'true':
        print('Training HenNet on GPU mode ...\n')
        time.sleep(1.0)
        hn = HenNet_GPU(c_pad=val_c_emb.shape[-1].c, h_pad=args.q, hidden_scale=args.d)
        H = hn.build_model()
        H.fit_generator(train_generator, validation_data=([val_h_emb, val_c_emb], [val_targets]), epochs=50, steps_per_epoch=len(train_generator),
                        shuffle=True, use_multiprocessing=True, workers=6, callbacks=[monitor_span(), checkpoint])

    return


main()