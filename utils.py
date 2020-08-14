import tensorflow as tf
import numpy as np
import os
from math import ceil
from glob import glob
from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.keras import backend as K

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3

def load_data(dirname):
    """Loads CIFAR10 dataset.

    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
      fpath = os.path.join(dirname, 'data_batch_' + str(i))
      (x_train[(i - 1) * 10000:i * 10000, :, :, :],
      y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    fpath = os.path.join(dirname, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
      x_train = x_train.transpose(0, 2, 3, 1)
      x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)

def get_best_model(model_dir, model='best_model'):
    model_to_restore = None
    list_best_model_index = glob(os.path.join(model_dir, '{}.ckpt-*.index'.format(model)))
    if len(list_best_model_index) > 0:
        model_to_restore = list_best_model_index[0].split('.index')[0]
    return model_to_restore

def map_func_train(image, label):

    image = tf.image.resize_image_with_crop_or_pad(image, HEIGHT + 8, WIDTH + 8)

    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.random_crop(image, [HEIGHT, WIDTH, NUM_CHANNELS])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)

    return image, label

def map_func_val_test(image, label):
    image = tf.image.per_image_standardization(image)
    return image, label

def build_dataset(sess, x_train_np, x_val_np, x_test_np, y_train_np, y_val_np, y_test_np,
                  batch_size=32,
                  buffer_size=40000):

    with tf.variable_scope('DATA'):

        with tf.name_scope('dataset_placeholders'):
            x_train_tf = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32, name='X_train')
            x_val_tf = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32, name='X_val')
            x_test_tf = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32, name='X_test')

            y_train_tf = tf.placeholder(shape=[None, ], dtype=tf.int64, name='y_train')
            y_val_tf = tf.placeholder(shape=[None, ], dtype=tf.int64, name='y_val')
            y_test_tf = tf.placeholder(shape=[None, ], dtype=tf.int64, name='y_test')

        with tf.name_scope('dataset_train'):
            dataset_train = tf.data.Dataset.from_tensor_slices((x_train_tf, y_train_tf))
            dataset_train = dataset_train.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
            dataset_train = dataset_train.repeat()
            dataset_train = dataset_train.map(map_func_train, num_parallel_calls=os.cpu_count()//2)
            dataset_train = dataset_train.batch(batch_size=batch_size)

        with tf.name_scope('dataset_val'):
            dataset_val = tf.data.Dataset.from_tensor_slices((x_val_tf, y_val_tf))
            dataset_val = dataset_val.map(map_func_val_test, num_parallel_calls=os.cpu_count()//2)
            dataset_val = dataset_val.batch(batch_size=batch_size)
            dataset_val = dataset_val.repeat()

        with tf.name_scope('dataset_test'):
            dataset_test = tf.data.Dataset.from_tensor_slices((x_test_tf, y_test_tf))
            dataset_test = dataset_test.map(map_func_val_test, num_parallel_calls=os.cpu_count()//2)
            dataset_test = dataset_test.batch(batch_size=batch_size)
            dataset_test = dataset_test.repeat()

        with tf.name_scope('iterators'):
            handle = tf.placeholder(name='handle', shape=[], dtype=tf.string)
            iterator = tf.data.Iterator.from_string_handle(handle, dataset_train.output_types, dataset_train.output_shapes)
            batch_x, batch_y = iterator.get_next()

            iterator_train = dataset_train.make_initializable_iterator()
            iterator_val = dataset_val.make_initializable_iterator()
            iterator_test = dataset_test.make_initializable_iterator()

        handle_train = sess.run(iterator_train.string_handle())
        handle_val = sess.run(iterator_val.string_handle())
        handle_test = sess.run(iterator_test.string_handle())

        print('...initialize datasets...')
        sess.run(iterator_train.initializer, feed_dict={x_train_tf: x_train_np, y_train_tf: y_train_np})
        sess.run(iterator_val.initializer, feed_dict={x_val_tf: x_val_np, y_val_tf: y_val_np})
        sess.run(iterator_test.initializer, feed_dict={x_test_tf: x_test_np, y_test_tf: y_test_np})

    return batch_x, batch_y, handle, handle_train, handle_val, handle_test
