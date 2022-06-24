import argparse
import tensorflow as tf

import time

parser = argparse.ArgumentParser(description='Tensorflow 2.0 MNIST Elastic Autoscaling Example')

parser.add_argument('--batch-size', type=int, default=64,
                    help='number of items per mini-batch')
parser.add_argument('--batches-per-epoch', type=int, default=10000,
                    help='the batches to do for an epoch')
parser.add_argument('--num-epochs', type=int, default=5,
                    help='number epochs to train')

args = parser.parse_args()


(mnist_images, mnist_labels), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

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
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
loss = tf.losses.SparseCategoricalCrossentropy()

lr = 0.001
opt = tf.optimizers.Adam(lr)


@tf.function
def training_step(images, labels):
    with tf.GradientTape() as tape:
        probs = mnist_model(images, training=True)
        loss_value = loss(labels, probs)

    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    opt.apply_gradients(zip(grads, mnist_model.trainable_variables))
    return loss_value


def train():
    for epoch in range(args.num_epochs):
        print("Epoch: {}".format(epoch))

        for batch_idx, (images, labels) in enumerate(dataset.take(args.batches_per_epoch)):
            training_step(images, labels)


print('Training starting')
start_time = time.time()
train()
training_time = time.time() - start_time
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

print("Model accuracy: %.6f" % (float(test_acc),))