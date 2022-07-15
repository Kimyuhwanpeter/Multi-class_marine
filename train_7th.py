# -*- coding:utf-8 -*-
from model_7th import *
from random import random
from model_profiler import model_profiler
from color_to_gray import *

import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_height": 256,
                           
                           "img_width": 320,

                           "classes": 8,
                           
                           "batch_size": 8,
                           
                           "epochs": 200,
                           
                           "lr": 0.0001,
                           
                           "tr_img_path": "D:/[1]DB/[5]4th_paper_DB/SUIM/train_val/images",
                           
                           "tr_lab_path": "D:/[1]DB/[5]4th_paper_DB/SUIM/train_val/masks",
                           
                           "te_img_path": "D:/[1]DB/[5]4th_paper_DB/SUIM/TEST/images",
                           
                           "te_lab_path": "D:/[1]DB/[5]4th_paper_DB/SUIM/TEST/masks_fix",

                           "train": True,
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "",
                           
                           "save_checkpoint_path": ""})

optim = tf.keras.optimizers.Adam(FLAGS.lr)

def train_func(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_height, FLAGS.img_width])
    img = tf.image.random_brightness(img, max_delta=50.)
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    img = tf.image.random_hue(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    img = tf.clip_by_value(img, 0, 255)
    img = tf.image.rgb_to_grayscale(img)
    img = img / 255.
    img = tf.concat([img, img, img, img, img, img, img, img], -1)

    lab = tf.io.read_file(lab_list)
    lab = tf.image.decode_bmp(lab)
    lab = tf.image.resize(lab, [FLAGS.img_height, FLAGS.img_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    if random() > 0.5:
        img = tf.image.flip_left_right(img)
        lab = tf.image.flip_left_right(lab)

    return img, lab

def test_func(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_height, FLAGS.img_width])
    img = tf.clip_by_value(img, 0, 255)
    img = tf.image.rgb_to_grayscale(img)
    img = img / 255.
    img = tf.concat([img, img, img, img, img, img, img, img], -1)

    lab = tf.io.read_file(lab_list)
    lab = tf.image.decode_bmp(lab)
    lab = tf.image.resize(lab, [FLAGS.img_height, FLAGS.img_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return img, lab

def cal_loss(model, images, labels):

    with tf.GradientTape() as tape:

        output = model(images, True)
        output = tf.reshape(output, [-1, FLAGS.classes])
        labels = tf.reshape(labels, [-1, FLAGS.classes])

        # 여기서부터 ranking loss로 접근하는것!!


    return loss

def main():
    model = Fix_model(input_shape=(FLAGS.img_height, FLAGS.img_width, FLAGS.classes), classes=FLAGS.classes)
    prob = model_profiler(model, FLAGS.batch_size)
    model.summary()
    print(prob)

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored!!!")

    if FLAGS.train:

        count = 0

        tr_img_data = os.listdir(FLAGS.tr_img_path)
        tr_img_data = [FLAGS.tr_img_path + "/" + data for data in tr_img_data]
        tr_img_data = np.array(tr_img_data)
        tr_lab_data = os.listdir(FLAGS.tr_lab_path)
        tr_lab_data = [FLAGS.tr_lab_path + "/" + data for data in tr_lab_data]
        tr_lab_data = np.array(tr_lab_data)

        te_img_data = os.listdir(FLAGS.te_img_path)
        te_img_data = [FLAGS.te_img_path + "/" + data for data in te_img_data]
        te_img_data = np.array(te_img_data)
        te_lab_data = os.listdir(FLAGS.te_lab_path)
        te_lab_data = [FLAGS.te_lab_path + "/" + data for data in te_lab_data]
        te_lab_data = np.array(te_lab_data)

        te_gener = tf.data.Dataset.from_tensor_slices((te_img_data, te_lab_data))
        te_gener = te_gener.map(test_func)
        te_gener = te_gener.batch(1)
        te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)

        for epoch in range(FLAGS.epochs):

            tr_gener = tf.data.Dataset.from_tensor_slices((tr_img_data, tr_lab_data))
            tr_gener = tr_gener.shuffle(len(tr_img_data))
            tr_gener = tr_gener.map(train_func)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_iter = iter(tr_gener)
            tr_idx = len(tr_img_data) // FLAGS.batch_size
            for step in range(tr_idx):
                temp_batch_labels = tf.zeros([FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width], tf.float32)
                batch_images, batch_labels = next(tr_iter)
                
                labels = tf.where(func22(batch_labels) & func23(batch_labels) & func24(batch_labels), 7, temp_batch_labels)
                labels = tf.where(func19(batch_labels) & func20(batch_labels) & func21(batch_labels), 6, labels)
                labels = tf.where(func16(batch_labels) & func17(batch_labels) & func18(batch_labels), 5, labels)
                labels = tf.where(func13(batch_labels) & func14(batch_labels) & func15(batch_labels), 4, labels)
                labels = tf.where(func10(batch_labels) & func11(batch_labels) & func12(batch_labels), 3, labels)
                labels = tf.where(func7(batch_labels) & func8(batch_labels) & func9(batch_labels), 2, labels)
                labels = tf.where(func4(batch_labels) & func5(batch_labels) & func6(batch_labels), 1, labels)
                labels = tf.where(func1(batch_labels) & func2(batch_labels) & func3(batch_labels), 0, labels)

                labels = tf.one_hot(labels, FLAGS.classes)

                loss = cal_loss(model, batch_images, labels)




        


if __name__ == "__main__":
    main()

