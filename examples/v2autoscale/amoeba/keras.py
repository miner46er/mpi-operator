import time
import tensorflow as tf
import horovod.tensorflow.keras as hvd


class AutoscaleKerasCallback(tf.keras.callbacks.Callback):
    def __init__(self, autoscaler, batch_size):
        self.autoscaler = autoscaler
        self.batch_size = batch_size
        self.batch_start = None

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start = time.time()

    def on_train_batch_end(self, batch, logs=None):
        self.autoscaler.adjust_scaling(self.batch_size * hvd.size() / (time.time() - self.batch_start))
