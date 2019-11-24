'''TensorFlow implementation of http://arxiv.org/pdf/1312.6114v10.pdf'''

from __future__ import absolute_import, division, print_function
from sklearn.model_selection import KFold
import math
import os

import numpy as np
import prettytensor as pt
import random
import scipy.misc
import tensorflow as tf
#from scipy.misc import imsave
#from scipy.misc import imread
from tensorflow.examples.tutorials.mnist import input_data

#from deconv import deconv2d
from progressbar import ETA, Bar, Percentage, ProgressBar

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("input_size", 32000, "size of input")
flags.DEFINE_integer("batch_size", 20, "batch size")
flags.DEFINE_integer("updates_per_epoch", 8, "number of updates per epoch") #put all the data into sinlge batch
flags.DEFINE_integer("max_epoch", 200, "max epoch")
flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_integer("hidden_size", 200, "size of the hidden VAE unit")

FLAGS = flags.FLAGS

def sigmoid(x):
  return 1 / (1 + np.exp(-5*x))

def encoder(input_tensor):
    '''Create encoder network.

    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28]

    Returns:
        A tensor that expresses the encoder network
    '''
    return (pt.wrap(input_tensor).
            reshape([None, 1, 1, FLAGS.input_size]).
            flatten().
            fully_connected(500, activation_fn=tf.nn.relu, name='enc_weight_1').         #   dropout(0.9).
            fully_connected(FLAGS.hidden_size * 2, activation_fn=None, name='enc_weight_2')).tensor

def encoder_test(input_tensor):
    '''Create encoder network.

    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28]

    Returns:
        A tensor that expresses the encoder network
    '''
    return (pt.wrap(input_tensor).
            reshape([None, 1, 1, FLAGS.input_size]).
            flatten().
            fully_connected(500, activation_fn=tf.nn.relu, name='enc_weight_1').
            fully_connected(FLAGS.hidden_size * 2, activation_fn=None, name='enc_weight_2')).tensor



def decoder(input_tensor=None):
    '''Create decoder network.

        If input tensor is provided then decodes it, otherwise samples from
        a sampled vector.
    Args:
        input_tensor: a batch of vectors to decode

    Returns:
        A tensor that expresses the decoder network
    '''
    epsilon = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size])
    if input_tensor is None:
        mean = None
        stddev = None
        input_sample = epsilon
    else:
        mean = input_tensor[:, :FLAGS.hidden_size]
        stddev = tf.sqrt(tf.exp(input_tensor[:, FLAGS.hidden_size:]))
        input_sample = mean + epsilon * stddev
    return (pt.wrap(input_sample).
            reshape([None, FLAGS.hidden_size]).
            fully_connected(500, activation_fn=tf.nn.relu, name='dec_weight_1').
            fully_connected(FLAGS.input_size, activation_fn=None, name='dec_weight_2').
            flatten()).tensor, mean, stddev

def decoder_test(input_tensor=None):

    input_sample = input_tensor[:, :FLAGS.hidden_size]

    return (pt.wrap(input_sample).
            reshape([None, FLAGS.hidden_size]).
            fully_connected(500, activation_fn=tf.nn.relu, name='dec_weight_1').
            fully_connected(FLAGS.input_size, activation_fn=None, name='dec_weight_2').
            flatten()).tensor, mean, stddev



def get_vae_cost(mean, stddev, epsilon=1e-8):
    '''VAE loss
        See the paper

    Args:
        mean:
        stddev:
        epsilon:
    '''
    return tf.reduce_sum(0.5 * (tf.square(mean) + tf.square(stddev) -
                                2.0 * tf.log(stddev + epsilon) - 1.0))


def get_reconstruction_cost(output_tensor, target_tensor):
    '''Reconstruction loss

    Cross entropy reconstruction loss

    Args:
        output_tensor: tensor produces by decoder
        target_tensor: the target tensor that we want to reconstruct
        epsilon:
    '''
    return tf.reduce_sum(tf.nn.l2_loss(output_tensor-target_tensor))
    #return tf.reduce_sum(-target_tensor * tf.log(output_tensor + epsilon) -
    #                     (1.0 - target_tensor) * tf.log(1.0 - output_tensor + epsilon))


def generate_train_batch(textData, cnt, dim):
    x = np.random.permutation(textData)
    g = np.copy(x)

    if cnt < 100:

        return x, g

    else:
        #add missing from the data
        r = get_random_missing(dim, 28800)
        x = x*r
        return x, g


def generate_test_batch(testData, dim, cnt, missing):

    if cnt < 100:
        return testData
    else:
        r = get_random_missing(dim, missing)
        x = testData*r

        return x

def get_random_missing(dim, percent):

    x = np.zeros([dim, FLAGS.input_size], dtype=np.float32)

    for ind in range(0, dim):
        r = np.random.choice(FLAGS.input_size, percent) # missing
        x[ind, r] = 1

    return x

def validation_label(textData, rand_order, random_fold, fold): # variable fold means fold info

    testset = np.zeros([20, FLAGS.input_size], dtype=np.float32)
    trainset = np.zeros([140, FLAGS.input_size], dtype=np.float32)
    label_test = np.zeros([20, 1], dtype=np.float32)
    label_train = np.zeros([140, 1], dtype=np.float32)



    current_fold = random_fold[fold]
    for k in range(0,20):
        current_fold[k] = int(current_fold[k])

    # for testset
    for i in range(0,20):
        testset[i,:] = textData[int(current_fold[i]),:FLAGS.input_size]
        label_test[i,0] = textData[int(current_fold[i]), FLAGS.input_size]

    # for trainset
    cnt = 0
    for j in range(0,160):
        if j in current_fold:
            j=j
        else:
            trainset[cnt,:] = textData[int(rand_order[j]),:FLAGS.input_size]
            label_train[cnt, 0] = textData[int(rand_order[j]), FLAGS.input_size]
            cnt = cnt + 1

    '''
    test_range = range(cnt*FLAGS.batch_size, (cnt+1)*FLAGS.batch_size)

    cnt = 0
    unt = 0
    for ind in range(0, 160):
        if ind in test_range:
            testset[cnt, :] = textData[int(rand_order[ind]),:FLAGS.input_size]
            label_test[cnt, 0] = textData[int(rand_order[ind]), FLAGS.input_size]
            cnt = cnt + 1
        else:
            trainset[unt, :] = textData[int(rand_order[ind]), :FLAGS.input_size]
            label_train[unt, 0] = textData[int(rand_order[ind]), FLAGS.input_size]
            unt = unt + 1
    '''
    return testset, trainset, label_test, label_train


if __name__ == "__main__":
    train_foldername = "save"
    rand_foldername  = "rand"

    input_tensor = tf.placeholder(tf.float32, [None, FLAGS.input_size])
    te_input_tensor = tf.placeholder(tf.float32, [None, FLAGS.input_size])
    gt_tensor = tf.placeholder(tf.float32, [None, FLAGS.input_size])

    rand_folder = os.path.join('/home/cheeze/PycharmProjects/KJW/research/electric_nose', rand_foldername)

    # Create and load random number(size 160)
    if not os.path.exists(rand_folder):
        os.makedirs(rand_folder)
        rand_order = np.random.permutation(160)
        np.savetxt(os.path.join(rand_folder, 'permutation.txt'), rand_order)
    else:
        rand_order = np.loadtxt(os.path.join(rand_folder, 'permutation.txt'))

    rand_order = sorted(rand_order)
    # Load dataset 160*320001
    textData = np.loadtxt('/home/cheeze/PycharmProjects/KJW/research/electric_nose/ori_e_nose_data.dat')

    tr_data = textData[:, :32000]
    tr_label = textData[:, 32000]



    # Network and train environments
    with pt.defaults_scope(activation_fn=tf.nn.elu,
                           batch_normalize=True,
                           learned_moments_update_rate=0.0003,
                           variance_epsilon=0.001,
                           scale_after_normalization=True):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("model") as scope:
                output_tensor, mean, stddev = decoder(encoder(input_tensor))

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("model", reuse=True) as scope:
                sampled_tensor, _, _ = decoder()
            with tf.variable_scope("model", reuse=True) as scope:
                te_output_tensor, _, _ = decoder_test(encoder_test(te_input_tensor))
            with tf.variable_scope("model", reuse=True) as scope:
                latent_tensor = encoder_test(te_input_tensor)


    # Calculate the loss
    vae_loss = get_vae_cost(mean, stddev)
    rec_loss = get_reconstruction_cost(output_tensor, gt_tensor)

    loss = rec_loss + vae_loss

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
    train = pt.apply_optimizer(optimizer, losses=[loss])

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    # Set the random fold order
    random_fold = []
    #for i in range(0,8):
    #    temp_rand_fold = rand_order[i*20:i*20+20]
    #    random_fold.append(temp_rand_fold)

    # For iter fold
    for iter in range(0,10):
        iter_fold = '/home/cheeze/PycharmProjects/KJW/research/electric_nose/iter%d'%(iter+1)
        if not os.path.exists(iter_fold):
            os.makedirs(iter_fold)

        rand_order = np.random.permutation(160)
        seed = random.seed()
        kf = KFold(n_splits=8, shuffle=True)
        train_random_fold = []
        for idx, (train_idx, test_idx) in enumerate(kf.split(tr_data, tr_label)):
            print(idx, "train idx : ", train_idx)
            print(idx, "test idx: ", test_idx)

            random_fold.append(test_idx) # define random fold test data
            train_rand_order =  np.concatenate((train_idx, test_idx), axis = None)
            train_random_fold.append(train_rand_order)

        '''
        train_random_fold saves set of train data
        train_rand_order saves set of test data
        '''
        # For set fold
        for set in range(0,8):
            # Create directories
            set_fold = iter_fold + '/set%d'%(set+1)
            if not os.path.exists((set_fold)):
                os.makedirs(set_fold)

            # apply data
            testData, trainData, label_test, label_train = validation_label(textData, train_random_fold[set],random_fold, set)

            # Start train
            with tf.Session() as sess:
                sess.run(init)

                for epoch in range(FLAGS.max_epoch):
                    training_loss = 0.0

                    widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
                    pbar = ProgressBar(maxval=FLAGS.updates_per_epoch, widgets=widgets)
                    pbar.start()
                    for i in range(FLAGS.updates_per_epoch - 1):
                        pbar.update(i)
                        x, g = generate_train_batch(
                            trainData[i * FLAGS.batch_size:(i+1) * FLAGS.batch_size, :FLAGS.input_size], epoch, FLAGS.batch_size
                        )
                        tx = generate_test_batch(testData, 20, 130, 16000)
                        _, loss_value = sess.run([train, loss], {input_tensor: x, gt_tensor: g, te_input_tensor: tx})
                        training_loss += loss_value

                    training_loss = training_loss / \
                                    (FLAGS.updates_per_epoch * 32000 * FLAGS.batch_size)

                    print("Loss %f" % training_loss)

                # Apply original data and save
                tr = generate_test_batch(trainData[:, :FLAGS.input_size], 140, 0, 32000)
                te = generate_test_batch(testData, 20, 0, 32000)
                np.savetxt(os.path.join(set_fold, 'train_iter'+str(iter+1)+'_set'+str(set+1)+'.txt'), tr)
                np.savetxt(os.path.join(set_fold, 'test_iter'+str(iter+1)+'_set'+str(set+1)+'.txt'), te)
                np.savetxt(os.path.join(set_fold, 'test_iter'+str(iter+1)+'_set'+str(set+1)+'label_te.txt'), label_test)
                np.savetxt(os.path.join(set_fold, 'test_iter'+str(iter+1)+'_set'+str(set+1)+'label_tr.txt'), label_train)

                loss_index = 10
                for i in range(1,10): # Set the missing rate
                    missing_rate = (95 - i*5)/100

                    # Apply missing data
                    trm = generate_test_batch(trainData[:,:FLAGS.input_size], 140, 130, int(32000 * missing_rate))
                    tem = generate_test_batch(testData, 20, 130, int(32000 * missing_rate))

                    np.savetxt(os.path.join(set_fold, 'train_iter'+str(iter+1)+'_set'+str(set+1)+'_loss'+str(loss_index)+'trm.txt'), trm)

                    reconstructed_val_te, latent_val_te = sess.run([te_output_tensor, latent_tensor], {te_input_tensor: te})
                    reconstructed_val_tem, latent_val_tem = sess.run([te_output_tensor, latent_tensor], {te_input_tensor: tem})
                    reconstructed_val_tr, latent_val_tr = sess.run([te_output_tensor, latent_tensor], {te_input_tensor: tr})
                    reconstructed_val_trm, latent_val_trm = sess.run([te_output_tensor, latent_tensor], {te_input_tensor: trm})

                    np.savetxt(os.path.join(set_fold, 'test_iter'+str(iter+1)+'_set'+str(set+1)+'_loss'+str(loss_index)+'.txt'), tem)
                    np.savetxt(os.path.join(set_fold, 'test_iter'+str(iter+1)+'_set'+str(set+1)+'_loss'+str(loss_index)+'_recon_DAE.txt'), reconstructed_val_tem)
                    np.savetxt(os.path.join(set_fold, 'test_iter'+str(iter+1)+'_set'+str(set+1)+'_loss'+str(loss_index)+'latent_te.txt'), latent_val_te)
                    np.savetxt(os.path.join(set_fold, 'test_iter'+str(iter+1)+'_set'+str(set+1)+'_loss'+str(loss_index)+'latent_tem.txt'), latent_val_tem)
                    np.savetxt(os.path.join(set_fold, 'test_iter'+str(iter+1)+'_set'+str(set+1)+'_loss'+str(loss_index)+'latent_tr.txt'), latent_val_tr)
                    np.savetxt(os.path.join(set_fold, 'test_iter'+str(iter+1)+'_set'+str(set+1)+'_loss'+str(loss_index)+'latent_trm.txt'),latent_val_trm)
                    loss_index = loss_index + 5















































