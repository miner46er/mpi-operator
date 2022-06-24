# Copyright 2021 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
from distutils.version import LooseVersion

import horovod.tensorflow.keras as hvd
import tensorflow as tf

from amoeba import AmoebaTensorflow, AutoscaleKerasCallback

parser = argparse.ArgumentParser(description='Tensorflow 2.0 Keras MNIST Example')

parser.add_argument('--use-mixed-precision', action='store_true', default=False,
                    help='use mixed precision for training')
parser.add_argument('--autoscale', action='store_true', default=False,
                    help='enables autoscaling')
parser.add_argument('--autoscale-min-worker', type=int, default=1,
                    help='minimum number of workers')
parser.add_argument('--autoscale-max-worker', type=int, default=None,
                    help='maximum number of workers')
parser.add_argument('--autoscale-delta-worker', type=int, default=1,
                    help='the number of workers to add or remove per scaling')
parser.add_argument('--autoscale-efficiency-threshold', type=float, default=0.5,
                    help='the threshold of efficiency for the autoscaling to trigger')

args = parser.parse_args()

if args.use_mixed_precision:
    if LooseVersion(tf.__version__) >= LooseVersion('2.4.0'):
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy('mixed_float16')
    else:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy)

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

(mnist_images, mnist_labels), _ = \
    tf.keras.datasets.cifar100.load_data(label_mode="fine", path='mnist-%d.npz' % hvd.rank())

tf.random.set_seed(13517119)
batch_size = 128
dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
     tf.cast(mnist_labels, tf.int64))
)
dataset = dataset.repeat().shuffle(10000).batch(batch_size)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Horovod: adjust learning rate based on number of GPUs.
lr = 0.001
scaled_lr = lr * hvd.size()
opt = tf.optimizers.Adam(scaled_lr)

# Horovod: add Horovod DistributedOptimizer.
opt = hvd.DistributedOptimizer(
    opt, backward_passes_per_step=1, average_aggregated_gradients=True)

# Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
# uses hvd.DistributedOptimizer() to compute gradients.
model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
              optimizer=opt,
              metrics=['accuracy'],
              experimental_run_tf_function=False)

# Horovod: initialize optimizer state, so we can synchronize across workers
# Keras has empty optimizer variables() for TF2:
# https://sourcegraph.com/github.com/tensorflow/tensorflow@v2.4.1/-/blob/tensorflow/python/keras/optimizer_v2/optimizer_v2.py#L351:10
model.fit(dataset, steps_per_epoch=1, epochs=1, callbacks=None)

state = hvd.elastic.KerasState(model, batch=0, epoch=0)


def on_state_reset():
    tf.keras.backend.set_value(state.model.optimizer.lr, lr * hvd.size())
    # Re-initialize, to join with possible new ranks
    state.model.fit(dataset, steps_per_epoch=1, epochs=1, callbacks=None)


state.register_reset_callbacks([on_state_reset])

callbacks = [
    hvd.elastic.UpdateEpochStateCallback(state),
    hvd.elastic.UpdateBatchStateCallback(state),
    hvd.elastic.CommitStateCallback(state),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))
    if args.autoscale:
        autoscaler = AmoebaTensorflow(min_worker_size=args.autoscale_min_worker,
                                      max_worker_size=args.autoscale_max_worker,
                                      delta_worker_size=args.autoscale_delta_worker,
                                      efficiency_lower_limit=args.autoscale_efficiency_threshold)
        callbacks.append(AutoscaleKerasCallback(autoscaler, batch_size))


# Train the model.
# Horovod: adjust number of steps based on number of GPUs.
@hvd.elastic.run
def train(state):
    state.model.fit(dataset, steps_per_epoch=500 // hvd.size(),
                    epochs=24 - state.epoch, callbacks=callbacks,
                    verbose=1)


train(state)
