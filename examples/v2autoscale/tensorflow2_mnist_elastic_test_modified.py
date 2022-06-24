# Copyright 2020 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Li1000cense is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse

import horovod.tensorflow as hvd
import tensorflow as tf
import time
from mpi4py import MPI

from amoeba import AmoebaTensorflow

parser = argparse.ArgumentParser(description='Tensorflow 2.0 MNIST Elastic Autoscaling Example')

parser.add_argument('--batch-size', type=int, default=128,
                    help='number of items per mini-batch')
parser.add_argument('--batches-per-epoch', type=int, default=10000,
                    help='the batches to do for an epoch')
parser.add_argument('--batches-per-commit', type=int, default=1,
                    help='number of batches per commit')
parser.add_argument('--num-epochs', type=int, default=24,
                    help='number epochs to train')
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
parser.add_argument('--autoscale-throughput-average-batches', type=int, default=1,
                    help="the number of batches' throughput to average for scaling")

args = parser.parse_args()

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

(mnist_images, mnist_labels), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % hvd.rank())

tf.random.set_seed(13517119)
dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
     tf.cast(mnist_labels, tf.int64))
)
dataset = dataset.repeat().shuffle(10000).batch(args.batch_size)

mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, [3, 3], activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
loss = tf.losses.SparseCategoricalCrossentropy()

# Horovod: adjust learning rate based on number of GPUs.
lr = 0.001
opt = tf.optimizers.Adam(lr * hvd.size())


@tf.function
def training_step(images, labels, allreduce=True):
    with tf.GradientTape() as tape:
        probs = mnist_model(images, training=True)
        loss_value = loss(labels, probs)

    # Horovod: add Horovod Distributed GradientTape.
    if allreduce:
        tape = hvd.DistributedGradientTape(tape)

    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    opt.apply_gradients(zip(grads, mnist_model.trainable_variables))
    return loss_value


# Horovod: initialize model and optimizer state, so we can synchronize across workers
for batch_idx, (images, labels) in enumerate(dataset.take(1)):
    training_step(images, labels, allreduce=False)

if args.autoscale:
    autoscaler = AmoebaTensorflow(min_worker_size=args.autoscale_min_worker,
                                  max_worker_size=args.autoscale_max_worker,
                                  delta_worker_size=args.autoscale_delta_worker,
                                  efficiency_lower_limit=args.autoscale_efficiency_threshold)


@hvd.elastic.run
def train(state):
    for state.epoch in range(state.epoch, args.num_epochs):
        print("Epoch: {}".format(state.epoch))
        start_batch = state.batch

        # Horovod: adjust number of steps based on number of GPUs.
        batches_train = (args.batches_per_epoch - state.batch) // hvd.size()

        sum_x_batches_time = 0
        cycle_batch = 0
        for batch_idx, (images, labels) in enumerate(dataset.skip(state.batch).take(batches_train)):
            cycle_batch += 1
            start_time = time.time()
            state.batch = start_batch + batch_idx
            loss_value = training_step(images, labels)

            if state.batch % 10 == 0 and hvd.local_rank() == 0:
                print('Batch #%d\tLoss: %.6f' % (state.batch, loss_value))

            if state.batch % args.batches_per_commit == 0:
                # Horovod: commit state every N batches
                state.commit()
            else:
                state.check_host_updates()

            end_time = time.time()
            if not args.autoscale:
                continue

            sum_x_batches_time += end_time - start_time
            if cycle_batch % args.autoscale_throughput_average_batches != 0:
                continue

            local_throughput_average = args.batch_size * args.autoscale_throughput_average_batches / sum_x_batches_time
            global_throughput = MPI.COMM_WORLD.allreduce(local_throughput_average, op=MPI.SUM)
            if hvd.rank() == 0:
                autoscaler.adjust_scaling(global_throughput)

            sum_x_batches_time = 0

        state.batch = 0


def on_state_reset():
    opt.lr.assign(lr * hvd.size())


state = hvd.elastic.TensorFlowKerasState(mnist_model, opt, batch=0, epoch=0, start_time=time.time())
state.register_reset_callbacks([on_state_reset])

train(state)

# checkpoint_dir = './checkpoints'
# checkpoint = tf.train.Checkpoint(model=mnist_model, optimizer=opt)
#
# # Horovod: save checkpoints only on worker 0 to prevent other workers from
# # corrupting it.
# if hvd.rank() == 0:
#     checkpoint.save(checkpoint_dir)

if hvd.rank() == 0:
    training_time = time.time() - state.start_time
    print('Training takes %.6f seconds' % training_time)

    print('Evaluating model on test data set')
    test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()


    @tf.function
    def test_step(x, y):
        test_logits = mnist_model(x, training=False)
        test_acc_metric.update_state(y, test_logits)


    test_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_test[..., tf.newaxis] / 255.0, tf.float32),
         tf.cast(y_test, tf.int64))
    )
    test_dataset = test_dataset.batch(args.batch_size)
    for x_batch_test, y_batch_test in test_dataset:
        test_step(x_batch_test, y_batch_test)
    test_acc = test_acc_metric.result()
    test_acc_metric.reset_states()
    print("Model accuracy: %.6f" % (float(test_acc),))
