import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *


def build_encoder(img_input_shape=(128,128,1), att_input_shape=2):
    input1 = Input(img_input_shape)
    input2 = Input(att_input_shape)
    label_ = Dense(128, activation='relu')(input2)
    label_ = Dense(128*128, activation='relu')(label_)
    label_ = Reshape((128, 128, 1))(label_)
    inputs = Concatenate(axis=-1)([input1, label_])

    conv1 = Conv2D(64, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(inputs)  # 128 -> 64
    bn1 = BatchNormalization()(conv1)
    conv2 = Conv2D(128, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(bn1)  # 64 -> 32
    bn2 = BatchNormalization()(conv2)
    conv3 = Conv2D(256, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(bn2)  #32 -> 16
    bn3 = BatchNormalization()(conv3)
    conv4 = Conv2D(512, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(bn3)  #16 -> 8
    bn4 = BatchNormalization()(conv4)
    model = Model([input1, input2], bn4)
    model.summary()
    return model

def build_att_decoder():
    inputs = Input((8, 8, 512))
    att_dconv1 = Conv2DTranspose(256, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(inputs)
    att_bn1 = BatchNormalization()(att_dconv1)
    att_dconv2 = Conv2DTranspose(128, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(att_bn1)
    att_bn2 = BatchNormalization()(att_dconv2)
    att_dconv3 = Conv2DTranspose(64, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(att_bn2)
    att_bn4 = BatchNormalization()(att_dconv3)
    att = Conv2DTranspose(1, 3, strides=(2, 2), padding='same', activation='sigmoid')(att_bn4)
    model = Model(inputs, att)
    model.summary()
    return model

def build_img_decoder():
    inputs = Input((8, 8, 512))
    img_dconv1 = Conv2DTranspose(256, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(inputs)
    img_bn1 = BatchNormalization()(img_dconv1)
    img_dconv2 = Conv2DTranspose(128, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(img_bn1)
    img_bn2 = BatchNormalization()(img_dconv2)
    img_dconv3 = Conv2DTranspose(64, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(img_bn2)
    img_bn3 = BatchNormalization()(img_dconv3)
    img = Conv2DTranspose(1, 3, strides=(2, 2), padding='same', activation='tanh')(img_bn3)
    model = Model(inputs, img)
    model.summary()
    return model

def build_discriminator(input_shape=(128,128,1)):
    inputs = Input(input_shape)
    conv1 = Conv2D(64, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.01))(inputs)  # 128 -> 64
    conv2 = Conv2D(128, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.01))(conv1)  # 64 -> 32
    conv3 = Conv2D(256, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.01))(conv2)  # 32 -> 16
    conv4 = Conv2D(512, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.01))(conv3)  # 16 -> 8
    flat = Flatten()(conv4)
    classified = Dense(2, activation='softmax')(flat)
    validation = Conv2D(64, 3, strides=(1, 1), padding='same')(conv4)
    validation = Conv2D(2, 3, strides=(1, 1), padding='same')(validation)
    validation = Softmax(axis=-1)(validation)
    model = Model(inputs, [validation, classified])
    model.summary()
    return model
