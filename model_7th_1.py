# -*- coding:utf-8 -*-
from model_profiler import model_profiler
import tensorflow as tf

Conv2D = tf.keras.layers.Conv2D
BatchNormalization = tf.keras.layers.BatchNormalization
ReLU = tf.keras.layers.ReLU
MaxPool2D = tf.keras.layers.MaxPool2D

def Seperate_class_block(input, filters, n):

    h_1x1 = Conv2D(filters=filters, kernel_size=1, use_bias=False)(input)
    h_1x1 = [tf.nn.sigmoid(BatchNormalization()(h_1x1[:, :, :, i:i+int(8*n)])) for i in range(0, filters, int(8*n))]

    h_3x3 = Conv2D(filters=filters, kernel_size=3, padding="same", use_bias=False)(input)
    h_3x3 = [BatchNormalization()(h_3x3[:, :, :, i:i+int(8*n)]) for i in range(0, filters, int(8*n))]

    h_5x5 = Conv2D(filters=filters, kernel_size=5, padding="same", use_bias=False)(input)
    h_5x5 = [BatchNormalization()(h_5x5[:, :, :, i:i+int(8*n)]) for i in range(0, filters, int(8*n))]

    split_h = [ReLU()((h_3x3[i] * h_1x1[i]) + h_5x5[i]) for i in range(8)]
    h = tf.concat(split_h, -1)

    return h, split_h

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
    h, _ = Seperate_class_block(h, 64, 1)

    h = Conv2D(filters=128, kernel_size=3, padding="same", use_bias=True, name="block2_conv1")(h)
    h = BatchNormalization()(h)

    h_ = [tf.nn.sigmoid(h[:, :, :, (128//classes)*i:(128//classes)*(i+1)]) for i in range(classes)]

    h = ReLU()(h)
    h = Conv2D(filters=128, kernel_size=3, padding="same", use_bias=True, name="block2_conv2")(h)
    h = BatchNormalization()(h)
    h = [h_[i] * h[:, :, :, (128//classes)*i:(128//classes)*(i+1)] for i in range(classes)]
    block_1 = h
    h = tf.concat(h, -1)
    h = ReLU()(h)

    h = MaxPool2D((2,2), 2)(h)
    h, _ = Seperate_class_block(h, 128, 2)

    h = Conv2D(filters=256, kernel_size=3, padding="same", use_bias=True, name="block3_conv1")(h)
    h = BatchNormalization()(h)

    h_ = [tf.nn.sigmoid(h[:, :, :, (256//classes)*i:(256//classes)*(i+1)]) for i in range(classes)]

    h = ReLU()(h)
    h = Conv2D(filters=256, kernel_size=3, padding="same", use_bias=True, name="block3_conv2")(h)
    h = BatchNormalization()(h)
    h = [h_[i] * h[:, :, :, (256//classes)*i:(256//classes)*(i+1)] for i in range(classes)]
    h = tf.concat(h, -1)

    h_ = [h[:, :, :, (256//classes)*i:(256//classes)*(i+1)] for i in range(classes)]

    h = ReLU()(h)
    h = Conv2D(filters=256, kernel_size=3, padding="same", use_bias=True, name="block3_conv3")(h)
    h = BatchNormalization()(h)
    h = [h_[i] + h[:, :, :, (256//classes)*i:(256//classes)*(i+1)] for i in range(classes)]
    block_2 = h
    h = tf.concat(h, -1)
    h = ReLU()(h)

    h = MaxPool2D((2,2), 2)(h)
    h, _ = Seperate_class_block(h, 256, 4)

    h = Conv2D(filters=512, kernel_size=3, padding="same", use_bias=True, name="block4_conv1")(h)
    h = BatchNormalization()(h)

    h_ = [tf.nn.sigmoid(h[:, :, :, (512//classes)*i:(512//classes)*(i+1)]) for i in range(classes)]

    h = ReLU()(h)
    h = Conv2D(filters=512, kernel_size=3, padding="same", use_bias=True, name="block4_conv2")(h)
    h = BatchNormalization()(h)
    h = [h_[i] * h[:, :, :, (512//classes)*i:(512//classes)*(i+1)] for i in range(classes)]
    h = tf.concat(h, -1)

    h_ = [h[:, :, :, (512//classes)*i:(512//classes)*(i+1)] for i in range(classes)]

    h = ReLU()(h)
    h = Conv2D(filters=512, kernel_size=3, padding="same", use_bias=True, name="block4_conv3")(h)
    h = BatchNormalization()(h)
    h = [h_[i] + h[:, :, :, (512//classes)*i:(512//classes)*(i+1)] for i in range(classes)]
    block_3 = h
    h = tf.concat(h, -1)
    h = ReLU()(h)

    h = MaxPool2D((2,2), 2)(h)
    h, _ = Seperate_class_block(h, 512, 8)

    h = Conv2D(filters=512, kernel_size=3, padding="same", use_bias=True, name="block5_conv1")(h)
    h = BatchNormalization()(h)

    h_ = [tf.nn.sigmoid(h[:, :, :, (512//classes)*i:(512//classes)*(i+1)]) for i in range(classes)]

    h = ReLU()(h)
    h = Conv2D(filters=512, kernel_size=3, padding="same", use_bias=True, name="block5_conv2")(h)
    h = BatchNormalization()(h)
    h = [h_[i] * h[:, :, :, (512//classes)*i:(512//classes)*(i+1)] for i in range(classes)]
    h = tf.concat(h, -1)

    h_ = [h[:, :, :, (512//classes)*i:(512//classes)*(i+1)] for i in range(classes)]

    h = ReLU()(h)
    h = Conv2D(filters=512, kernel_size=3, padding="same", use_bias=True, name="block5_conv3")(h)
    h = BatchNormalization()(h)
    h = [h_[i] + h[:, :, :, (512//classes)*i:(512//classes)*(i+1)] for i in range(classes)]
    block_4 = h
    h = tf.concat(h, -1)
    h = ReLU()(h)

    h = MaxPool2D((2,2), 2)(h)
    h, _ = Seperate_class_block(h, 512, 8)

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
    h = [tf.concat([h[:, :, :, (256//classes)*i:(256//classes)*(i+1)], block_4[i]], -1) for i in range(classes)]
    h = tf.concat(h, -1)
    h = ReLU()(h)

    h = Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h)
    h = BatchNormalization()(h)

    h_ = [tf.nn.sigmoid(h[:, :, :, (256//classes)*i:(256//classes)*(i+1)]) for i in range(classes)]

    h = ReLU()(h)
    h = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", use_bias=False)(h)
    h = BatchNormalization()(h)
    h = [h_[i] * h[:, :, :, (256//classes)*i:(256//classes)*(i+1)] for i in range(classes)]
    h = tf.concat(h, -1)
    h = ReLU()(h)
    h, _ = Seperate_class_block(h, 256, 4)

    h = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding="same", use_bias=False)(h)
    h = BatchNormalization()(h)
    h = [tf.concat([h[:, :, :, (128//classes)*i:(128//classes)*(i+1)], block_3[i]], -1) for i in range(classes)]
    h = tf.concat(h, -1)
    h = ReLU()(h)
    
    h = Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h)
    h = BatchNormalization()(h)

    h_ = [tf.nn.sigmoid(h[:, :, :, (128//classes)*i:(128//classes)*(i+1)]) for i in range(classes)]

    h = ReLU()(h)
    h = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", use_bias=False)(h)
    h = BatchNormalization()(h)
    h = [h_[i] * h[:, :, :, (128//classes)*i:(128//classes)*(i+1)] for i in range(classes)]
    h = tf.concat(h, -1)
    h = ReLU()(h)
    h, _ = Seperate_class_block(h, 128, 2)

    h = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding="same", use_bias=False)(h)
    h = BatchNormalization()(h)
    h = [tf.concat([h[:, :, :, (64//classes)*i:(64//classes)*(i+1)], block_2[i]], -1) for i in range(classes)]
    h = tf.concat(h, -1)
    h = ReLU()(h)

    h = Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(h)
    h = BatchNormalization()(h)

    h_ = [tf.nn.sigmoid(h[:, :, :, (64//classes)*i:(64//classes)*(i+1)]) for i in range(classes)]

    h = ReLU()(h)
    h = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", use_bias=False)(h)
    h = BatchNormalization()(h)
    h = [h_[i] * h[:, :, :, (64//classes)*i:(64//classes)*(i+1)] for i in range(classes)]
    h = tf.concat(h, -1)
    h = ReLU()(h)
    h, _ = Seperate_class_block(h, 64, 1)

    h = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding="same", use_bias=False)(h)
    h = BatchNormalization()(h)
    h = [tf.concat([h[:, :, :, (32//classes)*i:(32//classes)*(i+1)], block_1[i]], -1) for i in range(classes)]
    h = tf.concat(h, -1)
    h = ReLU()(h)

    h = Conv2D(filters=32, kernel_size=3, padding="same", use_bias=False)(h)
    h = BatchNormalization()(h)

    h_ = [tf.nn.sigmoid(h[:, :, :, (32//classes)*i:(32//classes)*(i+1)]) for i in range(classes)]

    h = ReLU()(h)
    h = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", use_bias=False)(h)
    h = BatchNormalization()(h)
    h = [h_[i] * h[:, :, :, (32//classes)*i:(32//classes)*(i+1)] for i in range(classes)]
    h = tf.concat(h, -1)
    h = ReLU()(h)
    h, _ = Seperate_class_block(h, 32, 0.5)

    h = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=4, strides=2, padding="same", use_bias=False)(h)
    h = BatchNormalization()(h)
    h = ReLU()(h)

    h = Conv2D(filters=16, kernel_size=3, padding="same", use_bias=False)(h)
    h = BatchNormalization()(h)

    h_ = [tf.nn.sigmoid(h[:, :, :, (16//classes)*i:(16//classes)*(i+1)]) for i in range(classes)]

    h = ReLU()(h)
    h = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", use_bias=False)(h)
    h = BatchNormalization()(h)
    h = [ReLU()(h_[i] * h[:, :, :, (16//classes)*i:(16//classes)*(i+1)]) for i in range(classes)]
    
    h = [Conv2D(filters=1, kernel_size=3, padding="same")(h[i]) for i in range(classes)]
    h = tf.concat(h, -1)



    return tf.keras.Model(inputs=inputs, outputs=h)

#m = model_7th_1()
#prob = model_profiler(m, 10)
#m.summary()
#print(prob)
