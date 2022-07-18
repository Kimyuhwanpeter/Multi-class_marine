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
                           
                           "save_checkpoint": "",
                           
                           "sample_images": ""})

optim = tf.keras.optimizers.Adam(FLAGS.lr)
color_map = np.array([[0,0,0], [0,0,255], [0, 255, 0], [0, 255, 255], [255, 0, 0], [255, 0, 255], [255, 255, 0], [255, 255, 255]], np.uint8)
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

def cal_loss(model, images, labels, imbal):

    with tf.GradientTape() as tape:

        output = model(images, True)
        output = tf.reshape(output, [-1, FLAGS.classes])
        labels = tf.reshape(labels, [-1, FLAGS.classes])

        # 0인지 아닌지, 1인지 아닌지, 2인지 아닌지...,...,...
        predict_0, labels_0 = output[:, 0], labels[:, 0]
        labels_0 = tf.where(labels_0 == 0, 1, 0)
        loss_0 = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_0, predict_0)
        
        predict_1, labels_1 = output[:, 1], labels[:, 1]
        labels_1 = tf.where(labels_1 == 1, 1, 0)
        loss_1 = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_1, predict_1)
        
        predict_2, labels_2 = output[:, 2], labels[:, 2]
        labels_2 = tf.where(labels_2 == 2, 1, 0)
        loss_2 = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_2, predict_2)
        
        predict_3, labels_3 = output[:, 3], labels[:, 3]
        labels_3 = tf.where(labels_3 == 3, 1, 0)
        loss_3 = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_3, predict_3)
        
        predict_4, labels_4 = output[:, 4], labels[:, 4]
        labels_4 = tf.where(labels_4 == 4, 1, 0)
        loss_4 = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_4, predict_4)
        
        predict_5, labels_5 = output[:, 5], labels[:, 5]
        labels_5 = tf.where(labels_5 == 5, 1, 0)
        loss_5 = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_5, predict_5)
        
        predict_6, labels_6 = output[:, 6], labels[:, 6]
        labels_6 = tf.where(labels_6 == 6, 1, 0)
        loss_6 = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_6, predict_6)
        
        predict_7, labels_7 = output[:, 7], labels[:, 7]
        labels_7 = tf.where(labels_7 == 7, 1, 0)
        loss_7 = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels_7, predict_7)

        total_loss = imbal[0]*loss_0 + imbal[1]*loss_1 + imbal[2]*loss_2 + imbal[3]*loss_3 \
            + imbal[4]*loss_4 + imbal[5]*loss_5 + imbal[6]*loss_6 + imbal[7]*loss_7 \
            + tf.keras.losses.CategoricalCrossentropy(from_logits=True)(labels, output)
        
    grads = tape.gradient(total_loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return total_loss

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
            tr_miou = 0.
            for step in range(tr_idx):
                temp_batch_labels = tf.zeros([FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width], tf.float32)
                batch_images, batch_labels = next(tr_iter)
                
                class_imbal_labels = batch_labels
                class_imbal_labels_buf = 0.
                for i in range(FLAGS.batch_size):
                    class_imbal_label = class_imbal_labels[i]
                    class_imbal_label = np.reshape(class_imbal_label, [FLAGS.img_height*FLAGS.img_width, ])
                    count_c_i_lab = np.bincount(class_imbal_label, minlength=FLAGS.classes)
                    class_imbal_labels_buf += count_c_i_lab
                class_imbal_labels_buf = (np.max(class_imbal_labels_buf / np.sum(class_imbal_labels_buf)) + 1 - (class_imbal_labels_buf / np.sum(class_imbal_labels_buf)))
                class_imbal_labels_buf = tf.nn.softmax(class_imbal_labels_buf[0:FLAGS.total_classes-1]).numpy()

                labels = tf.where(func22(batch_labels) & func23(batch_labels) & func24(batch_labels), 7, temp_batch_labels)
                labels = tf.where(func19(batch_labels) & func20(batch_labels) & func21(batch_labels), 6, labels)
                labels = tf.where(func16(batch_labels) & func17(batch_labels) & func18(batch_labels), 5, labels)
                labels = tf.where(func13(batch_labels) & func14(batch_labels) & func15(batch_labels), 4, labels)
                labels = tf.where(func10(batch_labels) & func11(batch_labels) & func12(batch_labels), 3, labels)
                labels = tf.where(func7(batch_labels) & func8(batch_labels) & func9(batch_labels), 2, labels)
                labels = tf.where(func4(batch_labels) & func5(batch_labels) & func6(batch_labels), 1, labels)
                labels = tf.where(func1(batch_labels) & func2(batch_labels) & func3(batch_labels), 0, labels)
                labels = tf.cast(labels, tf.uint8)

                labels = tf.one_hot(labels, FLAGS.classes)

                loss = cal_loss(model, batch_images, labels, class_imbal_labels_buf)

                if count % 10 == 0:
                    print("Epoch: {} [{}/{}] loss = {}".format(epoch, step+1, tr_idx, loss))

                if count % 100 == 0:

                    output = model(batch_images, False)

                    for i in range(FLAGS.batch_size):
                        predict = tf.nn.softmax(output[i], -1)
                        predict = tf.argmax(predict, -1)
                        predict = tf.cast(predict, tf.int32).numpy()
                        gt_images = tf.argmax(labels[i], -1)
                        gt_images = tf.cast(gt_images, tf.int32).numpy()

                        ground_images = color_map[gt_images]
                        predict_images = color_map[predict]

                        plt.imsave(FLAGS.sample_images + "/" + "{}_{}_predict.png".format(count, i), predict_images)
                        plt.imsave(FLAGS.sample_images + "/" + "{}_{}_gt.png".format(count, i), ground_images)

                temp_out = model(batch_images, False)
                temp_predict = tf.nn.softmax(temp_out[0], -1)
                temp_predict = tf.argmax(temp_predict, -1)
                temp_predict = tf.cast(temp_predict, tf.int32)
                temp_labels = tf.argmax(labels, -1)
                temp_labels = tf.cast(temp_labels, tf.int32)


                miou_, cm = Measurement(predict=temp_predict,
                                    label=temp_labels, 
                                    shape=[FLAGS.img_height*FLAGS.img_width, ], 
                                    total_classes=FLAGS.classes).MIOU()

                tr_miou += miou_

                count += 1

            print("mIoU (train) = %.4f" % (tr_miou / tr_idx))

            te_iter = iter(te_gener)
            te_idx = len(te_img_data)
            te_miou = 0.
            for j in range(te_idx):
                temp_batch_labels = tf.zeros([FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width], tf.float32)
                test_images, test_labels = next(te_iter)
                labels = tf.where(func22(test_labels) & func23(test_labels) & func24(test_labels), 7, temp_batch_labels)
                labels = tf.where(func19(test_labels) & func20(test_labels) & func21(test_labels), 6, labels)
                labels = tf.where(func16(test_labels) & func17(test_labels) & func18(test_labels), 5, labels)
                labels = tf.where(func13(test_labels) & func14(test_labels) & func15(test_labels), 4, labels)
                labels = tf.where(func10(test_labels) & func11(test_labels) & func12(test_labels), 3, labels)
                labels = tf.where(func7(test_labels) & func8(test_labels) & func9(test_labels), 2, labels)
                labels = tf.where(func4(test_labels) & func5(test_labels) & func6(test_labels), 1, labels)
                labels = tf.where(func1(test_labels) & func2(test_labels) & func3(test_labels), 0, labels)
                labels = tf.cast(labels, tf.int32)

                outputs = model(test_images, False)
                output = tf.nn.softmax(outputs[0], -1)
                output = tf.argmax(output, -1)
                output = tf.cast(output, tf.int32)
                label = labels[0]

                miou_, cm = Measurement(predict=output,
                                    label=label, 
                                    shape=[FLAGS.img_height*FLAGS.img_width, ], 
                                    total_classes=FLAGS.classes).MIOU()
                te_miou += miou_

            print("mIoU (test) = %.4f" % (te_miou / len(te_img_data)))

            model_dir = "%s/%s" % (FLAGS.save_checkpoint, epoch)
            if not os.path.isdir(model_dir):
                print("Make {} folder to store the weight!".format(epoch))
                os.makedirs(model_dir)
            ckpt = tf.train.Checkpoint(model=model, optim=optim)
            ckpt_dir = model_dir + "/7th__model_{}.ckpt".format(epoch)
            ckpt.save(ckpt_dir)
        


if __name__ == "__main__":
    main()

