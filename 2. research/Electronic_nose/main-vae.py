'''TensorFlow implementation of http://arxiv.org/pdf/1312.6114v10.pdf'''

from __future__ import absolute_import, division, print_function

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

def validation_label(textData, rand_order, cnt):

    testset = np.zeros([20, FLAGS.input_size], dtype=np.float32)
    trainset = np.zeros([140, FLAGS.input_size], dtype=np.float32)
    label_test = np.zeros([20, 1], dtype=np.float32)
    label_train = np.zeros([140, 1], dtype=np.float32)

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

    return testset, trainset, label_test, label_train





if __name__ == "__main__":

    train_foldername = "save"
    rand_foldername  = "rand"

    input_tensor = tf.placeholder(tf.float32, [None, FLAGS.input_size]) #input size 32000
    te_input_tensor = tf.placeholder(tf.float32, [None, FLAGS.input_size]) #input size 32000
    gt_tensor = tf.placeholder(tf.float32, [None, FLAGS.input_size]) #input size 32000

    rand_folder = os.path.join('/home/cheeze/PycharmProjects/KJW/research/electric_nose', rand_foldername)

    if not os.path.exists(rand_folder):
        os.makedirs(rand_folder)
        rand_order = np.random.permutation(160)
        np.savetxt(os.path.join(rand_folder, 'permutation.txt'), rand_order)
    else:
        rand_order = np.loadtxt(os.path.join(rand_folder, 'permutation.txt'))

    textData = np.loadtxt('/home/cheeze/PycharmProjects/KJW/research/electric_nose/ori_e_nose_data.dat')

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

    vae_loss = get_vae_cost(mean, stddev)
    rec_loss = get_reconstruction_cost(output_tensor, gt_tensor)

    loss = rec_loss + vae_loss

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
    train = pt.apply_optimizer(optimizer, losses=[loss])

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    for val in range(0, 8):
        print("do %d iteration" % val)

        # apply data
        testData, trainData, label_test, label_train = validation_label(textData, rand_order, val)

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
                        trainData[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size, :FLAGS.input_size], epoch, FLAGS.batch_size)
                    tx = generate_test_batch(testData, 20, 130, 16000)
                    _, loss_value = sess.run([train, loss], {input_tensor: x, gt_tensor: g, te_input_tensor: tx})
                    training_loss += loss_value

                training_loss = training_loss / \
                                (FLAGS.updates_per_epoch * 32000 * FLAGS.batch_size)

                print("Loss %f" % training_loss)

            save_folder = os.path.join('/home/cheeze/PycharmProjects/KJW/research/electric_nose', train_foldername)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
                save_path = saver.save(sess, os.path.join(save_folder, 'model.ckpt'))

            # test
            tr = generate_test_batch(trainData[:, :FLAGS.input_size], 140, 0, 32000) #original
            trm = generate_test_batch(trainData[:, :FLAGS.input_size], 140, 130, 16000) #missing
            te = generate_test_batch(testData, 20, 0, 32000)
            tem = generate_test_batch(testData, 20, 130, 16000)

            reconstructed_val_te, latent_val_te = sess.run([te_output_tensor, latent_tensor], {te_input_tensor: te})
            reconstructed_val_tem, latent_val_tem = sess.run([te_output_tensor, latent_tensor], {te_input_tensor: tem})
            reconstructed_val_tr, latent_val_tr = sess.run([te_output_tensor, latent_tensor], {te_input_tensor: tr})
            reconstructed_val_trm, latent_val_trm = sess.run([te_output_tensor, latent_tensor], {te_input_tensor: trm})

            np.savetxt(os.path.join(save_folder, 'reconstructed_te_'+str(val)+'.txt'), reconstructed_val_te)
            np.savetxt(os.path.join(save_folder, 'latent_te_'+str(val)+'.txt'), latent_val_te)
            np.savetxt(os.path.join(save_folder, 'reconstructed_tem_' + str(val) + '.txt'), reconstructed_val_tem)
            np.savetxt(os.path.join(save_folder, 'latent_tem_' + str(val) + '.txt'), latent_val_tem)
            np.savetxt(os.path.join(save_folder, 'reconstructed_tr_' + str(val) + '.txt'), reconstructed_val_tr)
            np.savetxt(os.path.join(save_folder, 'latent_tr_' + str(val) + '.txt'), latent_val_tr)
            np.savetxt(os.path.join(save_folder, 'reconstructed_trm_' + str(val) + '.txt'), reconstructed_val_trm)
            np.savetxt(os.path.join(save_folder, 'latent_trm_' + str(val) + '.txt'), latent_val_trm)
            np.savetxt(os.path.join(save_folder, 'label_te_' + str(val) + '.txt'), label_test)
            np.savetxt(os.path.join(save_folder, 'label_tr_' + str(val) + '.txt'), label_train)
            np.savetxt(os.path.join(save_folder, 'gt_te_'+str(val)+'.txt'), te)
            np.savetxt(os.path.join(save_folder, 'gt_tr_'+str(val)+'.txt'), tr)
            np.savetxt(os.path.join(save_folder, 'gt_tem_'+str(val)+'.txt'), tem)
            np.savetxt(os.path.join(save_folder, 'gt_trm_'+str(val)+'.txt'), trm)

