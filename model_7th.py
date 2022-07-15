# -*- coding:utf-8 -*-
from model_profiler import model_profiler

import tensorflow as tf

BatchNormalization = tf.keras.layers.BatchNormalization
LayerNormalization = tf.keras.layers.LayerNormalization
ReLU = tf.keras.layers.ReLU
Conv2D = tf.keras.layers.Conv2D
Conv2DTranspose = tf.keras.layers.Conv2DTranspose
DepthwiseConv2D = tf.keras.layers.DepthwiseConv2D
MaxPool2D = tf.keras.layers.MaxPool2D

concat = tf.keras.layers.concatenate

def Seperate_class_block(input, filters, n):

    h_1x1 = Conv2D(filters=filters, kernel_size=1, use_bias=False)(input)
    h_1x1 = [tf.nn.sigmoid(BatchNormalization()(h_1x1[:, :, :, i:i+8*n])) for i in range(0, filters, 8*n)]

    h_3x3 = Conv2D(filters=filters, kernel_size=3, padding="same", use_bias=False)(input)
    h_3x3 = [BatchNormalization()(h_3x3[:, :, :, i:i+8*n]) for i in range(0, filters, 8*n)]

    h_5x5 = Conv2D(filters=filters, kernel_size=5, padding="same", use_bias=False)(input)
    h_5x5 = [BatchNormalization()(h_5x5[:, :, :, i:i+8*n]) for i in range(0, filters, 8*n)]

    split_h = [ReLU()((h_3x3[i] * h_1x1[i]) + h_5x5[i]) for i in range(8)]
    h = concat(split_h, -1)

    return h, split_h

def Fix_model(input_shape=(256, 320, 8), classes=8):
    # 데이터 클래스에 너무 편향되지 않도록 모델을 설계하지 말자 (general 하게 만들고싶음)
    # 뭔가 혁신적인 방법은 없는걸까?
    # 4번째 5번째 6번쨰 논문에서 썼던 기법은 이제 그만, 사고력이 너무 좁아지는 느낌이다
    # 1. 입력 이미지는 그레이 스케일이고 8개의 채널로 우선 병합해서 사용한다
    temp_buf = []
    

    h = inputs = tf.keras.Input(input_shape)
    
    h = Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(h)
    h = [ReLU()(BatchNormalization()(h[:, :, :, i:i+classes])) for i in range(0, 64, classes)]
    h = concat(h, -1)

    h = DepthwiseConv2D(kernel_size=5, padding="same", use_bias=False)(h)
    h = [ReLU()(BatchNormalization()(h[:, :, :, i:i+classes])) for i in range(0, 64, classes)]
    h = concat(h, -1)

    h, _ = Seperate_class_block(h, 64, 1)

    h = MaxPool2D((2,2), 2)(h)

    h = Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h)
    h = [ReLU()(BatchNormalization()(h[:, :, :, i:i+classes*2])) for i in range(0, 128, classes*2)]
    h = concat(h, -1)

    h = DepthwiseConv2D(kernel_size=5, padding="same", use_bias=False)(h)
    h = [ReLU()(BatchNormalization()(h[:, :, :, i:i+classes*2])) for i in range(0, 128, classes*2)]
    h = concat(h, -1)

    h, block_1 = Seperate_class_block(h, 128, 2)


    h = MaxPool2D((2,2), 2)(h)

    h = Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h)
    h = [ReLU()(BatchNormalization()(h[:, :, :, i:i+classes*4])) for i in range(0, 256, classes*4)]
    h = concat(h, -1)

    h = Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h)
    h = [ReLU()(BatchNormalization()(h[:, :, :, i:i+classes*4])) for i in range(0, 256, classes*4)]
    h = concat(h, -1)

    h = DepthwiseConv2D(kernel_size=5, padding="same", use_bias=False)(h)
    h = [ReLU()(BatchNormalization()(h[:, :, :, i:i+classes*4])) for i in range(0, 256, classes*4)]
    h = concat(h, -1)

    h, block_2 = Seperate_class_block(h, 256, 4)

    h = MaxPool2D((2,2), 2)(h)

    h = Conv2D(filters=512, kernel_size=3, padding="same", use_bias=False)(h)
    h = [ReLU()(BatchNormalization()(h[:, :, :, i:i+classes*8])) for i in range(0, 512, classes*8)]
    h = concat(h, -1)

    h = Conv2D(filters=512, kernel_size=3, padding="same", use_bias=False)(h)
    h = [ReLU()(BatchNormalization()(h[:, :, :, i:i+classes*8])) for i in range(0, 512, classes*8)]
    h = concat(h, -1)

    h = DepthwiseConv2D(kernel_size=5, padding="same", use_bias=False)(h)
    h = [ReLU()(BatchNormalization()(h[:, :, :, i:i+classes*8])) for i in range(0, 512, classes*8)]
    h = concat(h, -1)

    h, block_3 = Seperate_class_block(h, 512, 8)

    h = MaxPool2D((2,2), 2)(h)

    h = Conv2D(filters=512, kernel_size=3, padding="same", use_bias=False)(h)
    h = [ReLU()(BatchNormalization()(h[:, :, :, i:i+classes*8])) for i in range(0, 512, classes*8)]
    h = concat(h, -1)

    h = DepthwiseConv2D(kernel_size=5, padding="same", use_bias=False)(h)
    h = [ReLU()(BatchNormalization()(h[:, :, :, i:i+classes*8])) for i in range(0, 512, classes*8)]
    h = concat(h, -1)

    ######################################################################################################

    h = Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding="same", use_bias=False)(h)
    h = [ReLU()(BatchNormalization()(h[:, :, :, i:i+classes*4])) for i in range(0, 256, classes*4)]
    h = [concat([h[i], block_3[i]], -1) for i in range(8)]
    h = concat(h, -1)

    h = Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h)
    h = [ReLU()(BatchNormalization()(h[:, :, :, i:i+classes*2])) for i in range(0, 128, classes*2)]
    h = concat(h, -1)

    h, _ = Seperate_class_block(h, 256, 4)

    h = Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding="same", use_bias=False)(h)
    h = [ReLU()(BatchNormalization()(h[:, :, :, i:i+classes*2])) for i in range(0, 128, classes*2)]
    h = [concat([h[i], block_2[i]], -1) for i in range(8)]
    h = concat(h, -1)

    h = Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(h)
    h = [ReLU()(BatchNormalization()(h[:, :, :, i:i+classes])) for i in range(0, 64, classes)]
    h = concat(h, -1)

    h, _ = Seperate_class_block(h, 128, 2)

    h = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding="same", use_bias=False)(h)
    h = [ReLU()(BatchNormalization()(h[:, :, :, i:i+classes])) for i in range(0, 64, classes)]
    h = [concat([h[i], block_1[i]], -1) for i in range(classes)]
    h = concat(h, -1)

    h = Conv2D(filters=32, kernel_size=3, padding="same", use_bias=False)(h)
    h = [ReLU()(BatchNormalization()(h[:, :, :, i:i+classes//2])) for i in range(0, 32, classes // 2)]
    h = concat(h, -1)

    h, _ = Seperate_class_block(h, 64, 1)

    h = Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding="same", use_bias=False)(h)
    h = [ReLU()(BatchNormalization()(h[:, :, :, i:i+classes//2])) for i in range(0, 32, classes // 2)]
    h = concat(h, -1)

    h = Conv2D(filters=32, kernel_size=3, padding="same", use_bias=False)(h)
    h = [ReLU()(BatchNormalization()(h[:, :, :, i:i+classes//2])) for i in range(0, 32, classes // 2)]

    h = [Conv2D(filters=1, kernel_size=3, padding="same", use_bias=False)(h[i]) for i in range(classes)]
    h = concat(h, -1)

    return tf.keras.Model(inputs=inputs, outputs=h)

#m = Fix_model()
#porb = model_profiler(m, 8)
#m.summary()
#print(porb)
