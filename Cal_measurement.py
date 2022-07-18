# -*- coding:utf-8 -*-
from color_to_gray import *
import tensorflow as tf
import numpy as np

class Measurement:
    def __init__(self, predict, label, shape, total_classes):
        self.predict = predict
        self.label = label
        self.total_classes = total_classes
        self.shape = shape

    def MIOU(self):
        # 0-crop, 1-weed, 2-background
        self.predict = np.reshape(self.predict, self.shape)
        #predict_ = np.array([self.predict[1], self.predict[0]])
        self.label = np.reshape(self.label, self.shape)
        #label_ = np.array([self.label[1], self.label[0]])

        predict_count = np.bincount(self.predict, minlength=self.total_classes)
        label_count = np.bincount(self.label, minlength=self.total_classes)

        temp = self.total_classes * np.array(self.label, dtype="int") + np.array(self.predict, dtype="int")  # Get category metrics

        temp_count = np.bincount(temp, minlength=self.total_classes * self.total_classes)
        cm = np.reshape(temp_count, [self.total_classes, self.total_classes])
        cm = np.diag(cm)

        U = label_count + predict_count - cm

        out = np.zeros((self.total_classes))
        miou = np.divide(cm, U, out=out, where=U != 0)

        indices = np.where(miou != 0)
        miou = np.take(miou, indices)

        miou = np.nanmean(miou)

        cm = tf.math.confusion_matrix(self.label, 
                                      self.predict,
                                      num_classes=self.total_classes).numpy()

        return miou, cm

#lab = tf.io.read_file("D:/[1]DB/[5]4th_paper_DB/SUIM/Test/masks_fix/d_r_58_.bmp")
#lab = tf.image.decode_bmp(lab)
#lab = tf.image.resize(lab, [256, 320], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#lab = tf.image.convert_image_dtype(lab, tf.uint8)
#lab = lab[tf.newaxis, :, :, :]

#temp_batch_labels = tf.zeros([1, 256, 320], tf.float32)
#labels = tf.where(func22(lab) & func23(lab) & func24(lab), 7, temp_batch_labels)
#labels = tf.where(func19(lab) & func20(lab) & func21(lab), 6, labels)
#labels = tf.where(func16(lab) & func17(lab) & func18(lab), 5, labels)
#labels = tf.where(func13(lab) & func14(lab) & func15(lab), 4, labels)
#labels = tf.where(func10(lab) & func11(lab) & func12(lab), 3, labels)
#labels = tf.where(func7(lab) & func8(lab) & func9(lab), 2, labels)
#labels = tf.where(func4(lab) & func5(lab) & func6(lab), 1, labels)
#labels = tf.where(func1(lab) & func2(lab) & func3(lab), 0, labels)
#labels = tf.cast(labels, tf.uint8)

#labels = labels[0]

#miou_, cm = Measurement(predict=labels,
#                    label=labels, 
#                    shape=[256*320, ],
#                    total_classes=8).MIOU()
#print(miou_)
