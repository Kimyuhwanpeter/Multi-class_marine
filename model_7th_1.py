# -*- coding:utf-8 -*-
from model_profiler import model_profiler
import tensorflow as tf

Conv2D = tf.keras.layers.Conv2D
BatchNormalization = tf.keras.layers.BatchNormalization
ReLU = tf.keras.layers.ReLU
MaxPool2D = tf.keras.layers.MaxPool2D


def model_7th_1(input_shape=(256, 320, 3), classes=8):

    h = inputs = tf.keras.Input(input_shape)

    h = Conv2D(filters=64, kernel_size=3, padding="same", use_bias=True, name="block1_conv1")(h)
    h = BatchNormalization()(h)

    h_ = [tf.nn.sigmoid(h[:, :, :, (64//classes)*i:(64//classes)*(i+1)]) for i in range(classes)]

    h = ReLU()(h)
    h = Conv2D(filters=64, kernel_size=3, padding="same", use_bias=True, name="block1_conv2")(h)
    h = BatchNormalization()(h)
    h = [h_[i] * h[:, :, :, (64//classes)*i:(64//classes)*(i+1)] for i in range(classes)]
    h = tf.concat(h, -1)
    h = ReLU()(h)

    h = MaxPool2D((2,2), 2)(h)

    h = Conv2D(filters=128, kernel_size=3, padding="same", use_bias=True, name="block2_conv1")(h)
    h = BatchNormalization()(h)

    h_ = [tf.nn.sigmoid(h[:, :, :, (128//classes)*i:(128//classes)*(i+1)]) for i in range(classes)]

    h = ReLU()(h)
    h = Conv2D(filters=128, kernel_size=3, padding="same", use_bias=True, name="block2_conv2")(h)
    h = BatchNormalization()(h)
    h = [h_[i] * h[:, :, :, (128//classes)*i:(128//classes)*(i+1)] for i in range(classes)]
    h = tf.concat(h, -1)
    h = ReLU()(h)

    h = MaxPool2D((2,2), 2)(h)

    h = Conv2D(filters=256, kernel_size=3, padding="same", use_bias=True, name="block3_conv1")(h)
    h = BatchNormalization()(h)

    h_ = [tf.nn.sigmoid(h[:, :, :, (256//classes)*i:(256//classes)*(i+1)]) for i in range(classes)]

    h = ReLU()(h)
    h = Conv2D(filters=256, kernel_size=3, padding="same", use_bias=True, name="block3_conv2")(h)
    h = BatchNormalization()(h)
    h = [h_[i] * h[:, :, :, (256//classes)*i:(256//classes)*(i+1)] for i in range(classes)]
    h = tf.concat(h, -1)

    h_ = [tf.nn.sigmoid(h[:, :, :, (256//classes)*i:(256//classes)*(i+1)]) for i in range(classes)]

    h = ReLU()(h)
    h = Conv2D(filters=256, kernel_size=3, padding="same", use_bias=True, name="block3_conv3")(h)
    h = BatchNormalization()(h)
    h = [h_[i] * h[:, :, :, (256//classes)*i:(256//classes)*(i+1)] for i in range(classes)]
    h = tf.concat(h, -1)
    h = ReLU()(h)

    h = MaxPool2D((2,2), 2)(h)

    h = Conv2D(filters=512, kernel_size=3, padding="same", use_bias=True, name="block4_conv1")(h)
    h = BatchNormalization()(h)

    h_ = [tf.nn.sigmoid(h[:, :, :, (512//classes)*i:(512//classes)*(i+1)]) for i in range(classes)]

    h = ReLU()(h)
    h = Conv2D(filters=512, kernel_size=3, padding="same", use_bias=True, name="block4_conv2")(h)
    h = BatchNormalization()(h)
    h = [h_[i] * h[:, :, :, (512//classes)*i:(512//classes)*(i+1)] for i in range(classes)]
    h = tf.concat(h, -1)

    h_ = [tf.nn.sigmoid(h[:, :, :, (512//classes)*i:(512//classes)*(i+1)]) for i in range(classes)]

    h = ReLU()(h)
    h = Conv2D(filters=512, kernel_size=3, padding="same", use_bias=True, name="block4_conv3")(h)
    h = BatchNormalization()(h)
    h = [h_[i] * h[:, :, :, (512//classes)*i:(512//classes)*(i+1)] for i in range(classes)]
    h = tf.concat(h, -1)
    h = ReLU()(h)

    h = MaxPool2D((2,2), 2)(h)

    h = Conv2D(filters=512, kernel_size=3, padding="same", use_bias=True, name="block5_conv1")(h)
    h = BatchNormalization()(h)

    h_ = [tf.nn.sigmoid(h[:, :, :, (512//classes)*i:(512//classes)*(i+1)]) for i in range(classes)]

    h = ReLU()(h)
    h = Conv2D(filters=512, kernel_size=3, padding="same", use_bias=True, name="block5_conv2")(h)
    h = BatchNormalization()(h)
    h = [h_[i] * h[:, :, :, (512//classes)*i:(512//classes)*(i+1)] for i in range(classes)]
    h = tf.concat(h, -1)

    h_ = [tf.nn.sigmoid(h[:, :, :, (512//classes)*i:(512//classes)*(i+1)]) for i in range(classes)]

    h = ReLU()(h)
    h = Conv2D(filters=512, kernel_size=3, padding="same", use_bias=True, name="block5_conv3")(h)
    h = BatchNormalization()(h)
    h = [h_[i] * h[:, :, :, (512//classes)*i:(512//classes)*(i+1)] for i in range(classes)]
    h = tf.concat(h, -1)
    h = ReLU()(h)

    h = MaxPool2D((2,2), 2)(h)

    h = Conv2D(filters=512, kernel_size=3, padding="same", use_bias=False)(h)
    h = BatchNormalization()(h)

    h_ = [tf.nn.sigmoid(h[:, :, :, (512//classes)*i:(512//classes)*(i+1)]) for i in range(classes)]

    h = ReLU()(h)
    h = Conv2D(filters=512, kernel_size=3, padding="same", use_bias=False)(h)
    h = BatchNormalization()(h)
    h = [h_[i] * h[:, :, :, (512//classes)*i:(512//classes)*(i+1)] for i in range(classes)]
    h = tf.concat(h, -1)
    h = ReLU()(h)

    #############3#############3#############3#############3#############3#############3#############3
    
    h = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding="same", use_bias=False)(h)
    h = BatchNormalization()(h)
    # encoder attention shape과 일치하지 않아서, 여기를 어떻게 해야한다

    return tf.keras.Model(inputs=inputs, outputs=h)

m = model_7th_1()
prob = model_profiler(m, 8)
m.summary()
print(prob)
