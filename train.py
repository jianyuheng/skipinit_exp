import tensorflow as tf
import numpy as np
import os
from math import ceil
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm
import argparse
import sys

from skipinit_resnet_model import Model
from utils import load_data, map_func_train, map_func_val_test, build_dataset, get_best_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MAIN_LOG_DIR = './'
INTERPOLATION = "BILINEAR"
data_dir = './cifar-10-batches-py'

PLOT_PERIOD = 20

DATA_FORMAT = "channels_first"
RESNET_SIZE = 32
BOTTLENECK = False
RESNET_VERSION = 2

COLORS = {
    'green': ['\033[32m', '\033[39m'],
    'red': ['\033[31m', '\033[39m']
}

def build_model(inputs, is_training_bn=True):

    num_blocks = (RESNET_SIZE - 2) // 6

    model = Model(resnet_size=RESNET_SIZE,
                  bottleneck=BOTTLENECK,
                  num_classes=10,
                  num_filters=16,
                  kernel_size=3,
                  conv_stride=1,
                  first_pool_size=None,
                  first_pool_stride=None,
                  block_sizes=[num_blocks, ]*3,
                  block_strides=[1, 2, 3],
                  final_size=64,
                  resnet_version=RESNET_VERSION,
                  data_format=DATA_FORMAT,
                  dtype=tf.float32)
    logits = model(inputs, training=is_training_bn)
    return logits


def main(params):

    log_dir = os.path.join(MAIN_LOG_DIR, params.log_dir)
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)

    tf.set_random_seed(seed=42)

    print("... creating a TensorFlow session ...\n")
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    print("... loading CIFAR10 dataset ...")
    (x_train, y_train), (x_test, y_test) = load_data(data_dir)
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    x_train, y_train = shuffle(x_train, y_train)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=0.2,
                                                      stratify=y_train,
                                                      random_state=51)
    # cast samples and labels
    x_train = x_train.astype(np.float32)
    x_val = x_val.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_val = y_val.astype(np.int32)
    y_test = y_test.astype(np.int32)

    print("\tTRAIN - images {} | {}  - labels {} - {}".format(x_train.shape, x_train.dtype, y_train.shape, y_train.dtype))
    print("\tVAL - images {} | {}  - labels {} - {}".format(x_val.shape, x_val.dtype, y_val.shape, y_val.dtype))
    print("\tTEST - images {} | {}  - labels {} - {}\n".format(x_test.shape, x_test.dtype, y_test.shape, y_test.dtype))

    print('... creating TensorFlow datasets ...\n')
    batch_x, batch_y, handle, handle_train, handle_val, handle_test \
        = build_dataset(sess, x_train, x_val, x_test, y_train, y_val, y_test,
                        batch_size=params.batch_size,
                        buffer_size=x_train.shape[0])

    nb_batches_per_epoch_train = int(ceil(x_train.shape[0]/params.batch_size))
    nb_batches_per_epoch_val = int(ceil(x_val.shape[0]/params.batch_size))
    nb_batches_per_epoch_test = int(ceil(x_test.shape[0]/params.batch_size))
    print('\tnb_batches_per_epoch_train : {}'.format(nb_batches_per_epoch_train))
    print('\tnb_batches_per_epoch_val : {}'.format(nb_batches_per_epoch_val))
    print('\tnb_batches_per_epoch_test : {}\n'.format(nb_batches_per_epoch_test))

    print('... building model ...\n')
    with tf.name_scope('INPUTS'):
        learning_rate = tf.placeholder(shape=[], dtype=tf.float32, name='learning_rate')
        is_training_bn = tf.placeholder(shape=[], dtype=tf.bool, name='is_training_bn')
        use_moving_statistics = tf.placeholder(shape=[], dtype=tf.bool, name='use_moving_statistics')
        global_step = tf.train.get_or_create_global_step()

    logits = build_model(batch_x, is_training_bn=is_training_bn)
    model_vars = tf.trainable_variables()

    with tf.name_scope('LOSS'):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=batch_y))
        acc = tf.reduce_mean(tf.cast(tf.equal(batch_y, tf.argmax(logits, axis=1)), dtype=tf.float32))

    with tf.name_scope('OPTIMIZER'):
        if params.opt == "adam":
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif params.opt == "momentum":
            opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=params.momentum)
        elif params.opt == "adamW":
            opt = tf.contrib.opt.AdamWOptimizer(weight_decay=params.weight_decay, learning_rate=learning_rate,
                                                beta1=0.9, beta2=0.999, epsilon=1e-8)
        elif params.opt == "momentumW":
            opt = tf.contrib.opt.MomentumWOptimizer(weight_decay=params.weight_decay, learning_rate=learning_rate,
                                                    momentum=params.momentum)
        else:
            raise ValueError('Invalid --opt argument : {}'.format(params.opt))

        if 'W' in params.opt:
            # when using AdamW or MomentumW
            if params.weight_decay_on == "all":
                decay_var_list = tf.trainable_variables()
            elif params.weight_decay_on == "kernels":
                decay_var_list = []
                for var in tf.trainable_variables():
                    if 'kernel' in var.name:
                        decay_var_list.append(var)
            else:
                raise ValueError('Invalid --weight_decay_on : {}'.format(params.weight_decay_on))

        # force updates of moving averages in BN before optimizing the network
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            grads_and_vars = opt.compute_gradients(loss, var_list=tf.trainable_variables())
            if 'W' in params.opt:
                # add decay_var_list argument for decoupled optimizers
                train_op = opt.apply_gradients(grads_and_vars, global_step=global_step,
                                               decay_var_list=decay_var_list, name='train_op')
            else:
                # without weight decay
                train_op = opt.apply_gradients(grads_and_vars, global_step=global_step, name='train_op')

    with tf.name_scope('METRICS'):
        acc_mean, acc_update_op = tf.metrics.mean(acc)
        loss_mean, loss_update_op = tf.metrics.mean(loss)

    # summaries which track loss/acc per batch
    acc_summary = tf.summary.scalar('TRAIN/acc', acc)
    loss_summary = tf.summary.scalar('TRAIN/loss', loss)

    # summaries which track accumulated loss/acc
    acc_mean_summary = tf.summary.scalar('MEAN/acc', acc_mean)
    loss_mean_summary = tf.summary.scalar('MEAN/loss', loss_mean)

    # summaries to plot at each epoch
    summaries_mean = tf.summary.merge([acc_mean_summary, loss_mean_summary], name='summaries_mean')

    # summaries to plot regularly
    summaries = [acc_summary, loss_summary]

    summaries = tf.summary.merge(summaries, name='summaries')

    with tf.name_scope('INIT_OPS'):
        # local init_ops contains operations to reset to zero accumulators of 'acc' and 'loss'
        global_init_op = tf.global_variables_initializer()
        local_init_op = tf.local_variables_initializer()

        sess.run(global_init_op)
        sess.run(local_init_op)

    with tf.name_scope('SAVERS'):
        best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    with tf.name_scope('FILE_WRITERS'):
        writer_train = tf.summary.FileWriter(os.path.join(log_dir, 'train'), graph=sess.graph)
        writer_val = tf.summary.FileWriter(os.path.join(log_dir, 'val'))
        writer_val_bn = tf.summary.FileWriter(os.path.join(log_dir, 'val_bn'))
        writer_test = tf.summary.FileWriter(os.path.join(log_dir, 'test'))

    if params.strategy_lr == "constant":
        def get_learning_rate(step, epoch, steps_per_epoch):
            return params.init_lr
    else:
        raise ValueError('Invalid --strategy_lr : {}'.format(params.strategy_lr))

    def inference(epoch, step, best_acc, best_step, best_epoch):
        sess.run(local_init_op)
        feed_dict = {is_training_bn: False, handle: handle_val}

        for _ in tqdm(range(nb_batches_per_epoch_val),
                      desc='VALIDATION @ EPOCH {}'.format(epoch)):
            sess.run([acc_update_op, loss_update_op], feed_dict=feed_dict)

        acc_v, loss_v, s = sess.run([acc_mean, loss_mean, summaries_mean])

        writer_val.add_summary(s, global_step=step)
        writer_val.flush()

        if acc_v > best_acc:
            color = COLORS['green']
            best_acc = acc_v
            best_step = step
            best_epoch = epoch
            ckpt_path = os.path.join(log_dir, 'best_model.ckpt')
            best_saver.save(sess, ckpt_path, global_step=step)
        else:
            color = COLORS['red']

        print("VALIDATION @ EPOCH {} : {}acc={:.4f}{}  loss={:.5f}".format(epoch, color[0], acc_v, color[1], loss_v))

        return best_acc, best_step, best_epoch

    feed_dict_train = {is_training_bn: True, handle: handle_train}

    best_acc = 0.
    best_step = 0
    best_epoch = 0

    step = -1

    # inference with trained variables
    best_acc, best_step, best_epoch = inference(0, 0, best_acc, best_step, best_epoch)

    for epoch in range(1, params.epochs+1):
        # ####################################### TRAIN 1 EPOCH ######################################################
        # re-initialize local variables
        sess.run(local_init_op)

        for _ in tqdm(range(nb_batches_per_epoch_train), desc='TRAIN @ EPOCH {}'.format(epoch)):
            step += 1

            feed_dict_train[learning_rate] = get_learning_rate(step, epoch, nb_batches_per_epoch_train)

            if step % PLOT_PERIOD == 0:
                _, s, _, _ = sess.run([train_op, summaries, acc_update_op, loss_update_op], feed_dict=feed_dict_train)
                writer_train.add_summary(s, global_step=step)

            else:
                sess.run([train_op, acc_update_op, loss_update_op], feed_dict=feed_dict_train)

        acc_v, loss_v, s = sess.run([acc_mean, loss_mean, summaries_mean])
        writer_train.add_summary(s, global_step=step)
        writer_train.flush()
        print("TRAIN @ EPOCH {} | : acc={:.4f}  loss={:.5f}".format(epoch, acc_v, loss_v))
        # ############################################################################################################

        # ###################################### INFERENCE ###########################################################
        # perform inference with trained variables and moving statistics
        best_acc, best_step, best_epoch = inference(epoch, step, best_acc, best_step, best_epoch)

    # Inference on test set with trained weights
    if best_acc > 0.:

        print("Load best model |  ACC={:.5f} form epoch={}".format(best_acc, best_epoch))
        model_to_restore = get_best_model(log_dir, model='best_model')
        if model_to_restore is not None:
            best_saver.restore(sess, model_to_restore)
        else:
            print("Impossible to load best model .... ")

        sess.run(local_init_op)
        feed_dict = {is_training_bn: False, handle: handle_test}
        for _ in tqdm(range(nb_batches_per_epoch_test), desc='TEST'):
            sess.run([acc_update_op, loss_update_op], feed_dict=feed_dict)

        acc_v, loss_v, s = sess.run([acc_mean, loss_mean, summaries_mean])
        writer_test.add_summary(s, global_step=best_step)
        writer_test.flush()
        print("TEST @ EPOCH {} : acc={:.4f}  loss={:.5f}".format(best_epoch, acc_v, loss_v))

    sess.close()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', dest='log_dir', type=str, default='model')

    parser.add_argument('--epochs', dest='epochs', type=int, default=170)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=128)

    parser.add_argument('--opt', dest='opt', type=str, default='momentumW') # adam, adamW, momentum, momentumW
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-4)
    parser.add_argument('--weight_decay_on', dest='weight_decay_on', type=str, default='all') # all or kernels

    parser.add_argument('--strategy_lr', dest='strategy_lr', type=str, default='constant') # constant

    parser.add_argument('--init_lr', dest='init_lr', type=float, default=0.01)
    params = parser.parse_args(sys.argv[1:])
    main(params)
