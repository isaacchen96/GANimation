import tensorflow as tf
from tensorflow.keras.losses import *

def w_distance(fake, real=None):
    if real is None:
        return tf.reduce_mean(fake)
    else:
        return tf.reduce_mean(fake) - tf.reduce_mean(real)

def adversarial_loss(pred, real=True):
    if real:
        gt = tf.concat([tf.ones((pred.shape[0], pred.shape[1], pred.shape[2], 1), dtype='float32'),
                   tf.zeros((pred.shape[0], pred.shape[1], pred.shape[2], 1), dtype='float32')], axis=-1)
    else:
        gt = tf.concat([tf.zeros((pred.shape[0], pred.shape[1], pred.shape[2], 1), dtype='float32'),
                   tf.ones((pred.shape[0], pred.shape[1], pred.shape[2], 1), dtype='float32')], axis=-1)
    mse = MeanSquaredError()
    return mse(gt, pred)

def gp(real_img, fake_img, discriminator):
    e = tf.random.uniform((len(real_img), 1, 1, 1), 0, 1)
    noise_img = e * real_img + (1. - e) * fake_img  # extend distribution space
    with tf.GradientTape() as tape:
        tape.watch(noise_img)
        o = discriminator(noise_img)
    g = tape.gradient(o, noise_img)  # image gradients
    g_norm2 = tf.sqrt(tf.reduce_sum(tf.square(g), axis=[1, 2, 3]))  # norm2 penalty
    gp = tf.square(g_norm2 - 1.)
    return tf.reduce_mean(gp)

def att_loss(att, lambda_=1000):
    loss = tf.reduce_sum(tf.square(att[:, :, :-1, :] - att[:, :, 1:, :])) + \
           tf.reduce_sum(tf.square(att[:, :-1, :, :] - att[:, 1:, :, :]))
    reg = tf.norm(att)
    return lambda_*loss , reg

def img_loss(x, y):
    mae = MeanAbsoluteError()
    return mae(x, y)

def classify_loss(gt, pred):

    return tf.reduce_mean(tf.square(gt-pred))
