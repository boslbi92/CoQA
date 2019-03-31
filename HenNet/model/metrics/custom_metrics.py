import numpy as np
import tensorflow as tf
import keras, os
from keras import backend as K
from keras.callbacks import Callback

# predict the most likely span
def get_best_span(span_begin_probs, span_end_probs):
    if len(span_begin_probs.shape) > 2 or len(span_end_probs.shape) > 2:
        raise ValueError("Input shapes must be (X,) or (1,X)")
    if len(span_begin_probs.shape) == 2:
        assert span_begin_probs.shape[0] == 1, "2D input must have an initial dimension of 1"
        span_begin_probs = span_begin_probs.flatten()
    if len(span_end_probs.shape) == 2:
        assert span_end_probs.shape[0] == 1, "2D input must have an initial dimension of 1"
        span_end_probs = span_end_probs.flatten()
    max_span_probability = 0
    best_word_span = (0, 1)
    begin_span_argmax = 0
    for j, _ in enumerate(span_begin_probs):
        val1 = span_begin_probs[begin_span_argmax]
        val2 = span_end_probs[j]

        if val1 * val2 > max_span_probability:
            best_word_span = (begin_span_argmax, j)
            max_span_probability = val1 * val2

        # We need to update best_span_argmax here _after_ we've checked the current span
        # position, so that we don't allow things like (1, 1), which are empty spans.  We've
        # added a special stop symbol to the end of the passage, so this still allows for all
        # valid spans over the passage.
        if val1 < span_begin_probs[j]:
            val1 = span_begin_probs[j]
            begin_span_argmax = j
    return (best_word_span[0], best_word_span[1])

class monitor_span(Callback):
    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        pred = np.array(self.model.predict([self.validation_data[0], self.validation_data[1]]))
        pred_start, pred_end = pred[:,0,:], pred[:,1,:]
        val_start, val_end = self.validation_data[2][:,0,:], self.validation_data[2][:,1,:]

        epoch = epoch + 1
        log_path = os.getcwd() + '/train_logs/'
        with open(log_path + "{}-trainlog.txt".format(epoch), 'w') as f:
            f.write('Prediction result on epoch {}\n\n'.format(epoch))
            for ps, pe, ts, te in zip(pred_start, pred_end, val_start, val_end):
                original_span = (np.argmax(ts, axis=0), np.argmax(te, axis=0))
                log_start, log_end = np.log(ps[original_span[0]]), np.log(ps[original_span[1]])
                result = (str(get_best_span(ps, pe)) + '\t' + str(original_span) + '\t' + str((log_start + log_end)) + '\n')
                f.write(result)
        f.close()
        return

def compute_loss(params):
    pred_start, pred_end, true_start_index, true_end_index = params[0], params[1], params[2], params[3]
    start_prob = pred_start[true_start_index]
    end_prob = pred_end[true_end_index]
    loss = K.log(start_prob) + K.log(end_prob)
    return loss

# custom loss function
def negative_log_span(y_true, y_pred):
    pred_start, pred_end = y_pred[:,0,:], y_pred[:,1,:]
    true_start, true_end = K.cast(K.argmax(y_true[:,0,:], axis=1), dtype='int32'), K.cast(K.argmax(y_true[:,1,:], axis=1), dtype='int32')
    batch_prob_sum = K.map_fn(compute_loss, elems=(pred_start, pred_end, true_start, true_end), dtype='float32')

    # debugging
    # y_true = tf.Print(y_true, [K.shape(true_start), (true_start, true_end)], "true_shape")
    # y_pred = tf.Print(y_pred, [K.shape(pred_end)], "pred_shape")

    return -K.mean(batch_prob_sum, axis=0)



