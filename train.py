# -*- coding: utf-8 -*-

import numpy as np
from keras.layers.core import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, Input, BatchNormalization, ReLU
from keras.models import Sequential, Model, load_model
from keras import backend as K
from keras.regularizers import l2
import os.path
import csv
import cv2 as cv
import glob
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json
from keras.callbacks import ModelCheckpoint, EarlyStopping
import math
from analysis import *
from matplotlib import pyplot

SEED = 13
batch_size = 32
shape = (128, 128, 3)
data_path = 'data/'


def horizontal_flip(img, degree):
    choice = np.random.choice([0, 1])  # 概率为 50%
    if choice == 1:
        img, degree = cv.flip(img, 1), -degree  # 图像水平翻转

    return img, degree


def random_brightness(img, degree):
    """
    随机调整输入图像的亮度，调整强度于 0.1（变黑）和 1（无变化）之间
    img: 输入图像
    degree: 输入图像对于的转动角度
    """
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    '''
    HSV 色彩空间：
        H - Hue 色调、色相，范围 0°-360°
            红色      黄色      绿色      青色      蓝色      品红      红色
            0°       60°      120°      180°     240°     300°      360°
        S - Saturation 饱和度，色彩的纯度，饱和度越低色彩越暗淡 0<=S<1
        V - Value 亮度，数值越高越接近于白色，数值越低越接近于黑色 0<=V<1
    '''
    alpha = np.random.uniform(low=0.1, high=1.0, size=None)  # 从一个均匀分布 [low,high) 中随机采样
    v = hsv[:, :, 2]
    v = v * alpha  # 调整亮度 V = alpha * V
    hsv[:, :, 2] = v.astype('uint8')
    rgb = cv.cvtColor(hsv.astype('uint8'), cv.COLOR_HSV2RGB)

    return rgb, degree


def left_right_random_swap(img_address, degree, degree_corr=1.0 / 4):
    """
    随机从左，中，右图像中选择一张图像，并相应调整转动的角度
    img_address: 中间图像的文件路径
    degree: 中间图像对于的方向盘转动角度
    degree_corr: 方向盘转动角度调整的值
    """
    swap = np.random.choice(['L', 'R', 'C'])

    if swap == 'L':
        img_address = img_address.replace('center', 'left')
        corrected_label = np.arctan(math.tan(degree) + degree_corr)
        return img_address, corrected_label

    elif swap == 'R':
        img_address = img_address.replace('center', 'right')
        corrected_label = np.arctan(math.tan(degree) - degree_corr)
        return img_address, corrected_label

    else:
        return img_address, degree


def discard_zero_steering(degrees, rate):
    """
    从角度为零的 index 中随机选择部分 index 返回
    degrees: 输入的角度值
    rate: 丢弃率，如果 rate=0.8，意味着 80% 的 index 会被返回，用于丢弃
    """
    steering_zero_idx = np.where(degrees == 0)
    '''
    np.where() 返回一个长度为 1 的元组，元素类型为 <class 'numpy.ndarray'> 为符合条件的数值的索引
    '''
    steering_zero_idx = steering_zero_idx[0]
    size_del = int(len(steering_zero_idx) * rate)
    idx_del = np.random.choice(steering_zero_idx, size=size_del, replace=False)
    '''
    replace 为是否可以取相同数字，在此处应当为 False
    '''
    return idx_del


def get_model(dropout_rate=0.1):

    input = Input(shape=shape)

    cv2d_1 = Conv2D(8, (3, 3), kernel_initializer='he_normal', input_shape=shape)(input)
    bn_1 = BatchNormalization()(cv2d_1)
    act_1 = ReLU()(bn_1)
    pool_1 = MaxPooling2D()(act_1)
    dropout_1 = Dropout(dropout_rate)(pool_1)

    cv2d_2 = Conv2D(8, (3, 3), kernel_initializer='he_normal')(dropout_1)
    bn_2 = BatchNormalization()(cv2d_2)
    act_2 = ReLU()(bn_2)
    pool_2 = MaxPooling2D()(act_2)
    dropout_2 = Dropout(dropout_rate)(pool_2)

    cv2d_3 = Conv2D(16, (3, 3), kernel_initializer='he_normal')(dropout_2)
    bn_3 = BatchNormalization()(cv2d_3)
    act_3 = ReLU()(bn_3)
    pool_3 = MaxPooling2D()(act_3)
    dropout_3 = Dropout(dropout_rate)(pool_3)

    cv2d_4 = Conv2D(16, (3, 3), kernel_initializer='he_normal')(dropout_3)
    bn_4 = BatchNormalization()(cv2d_4)
    act_4 = ReLU()(bn_4)
    pool_4 = MaxPooling2D()(act_4)
    dropout_4 = Dropout(dropout_rate)(pool_4)
    
    flatten = Flatten()(dropout_4)
    
    dense_1 = Dense(128, activation='relu')(flatten)
    dense_2 = Dense(32, activation='relu')(dense_1)
    dense_3 = Dense(8, activation='relu')(dense_2)
    output = Dense(1, activation='linear')(dense_3)

    model = Model(inputs=input, outputs=output)

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    return model


def image_transformation(img_address, degree):
    img_address, degree = left_right_random_swap(img_address, degree)
    img = cv.imread(img_address)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # opencv 读图像的色彩空间是 BGR 需要转换为 RGB
    img, degree = random_brightness(img, degree)
    img, degree = horizontal_flip(img, degree)

    return img, degree


def batch_generator(x, y, training=True):
    if training:  # 产生训练数据
        discard_rate = 1 - sum(y != 0) / 2 / sum(y == 0)
        rand_zero_idx = discard_zero_steering(y_train, rate=discard_rate)
        new_x = np.delete(x, rand_zero_idx, axis=0)
        new_y = np.delete(y, rand_zero_idx, axis=0)
        '''
        由于数据的严重不平衡性，所以把角度为 0 的训练数据随机丢弃
        numpy.delete() 此方法用于删除指定位置元素
        axis 用于指定哪个轴（通过 numpy.shape 查看）如果为 None 就是把数据拉平再删除，返回一个一维的 numpy 数组
        '''
        # print(len(new_y))
        # assert False
    else:  # 产生 validation 数据
        new_x = x
        new_y = y

    offset = 0
    while True:
        # 创建一个 batch 的数据
        X = np.empty((batch_size, *shape))
        Y = np.empty((batch_size, 1))

        for example in range(batch_size):
            img_address, img_steering = new_x[example + offset], new_y[example + offset]
            
            if training:
                img, img_steering = image_transformation(img_address, img_steering)
            else:
                img = cv.imread(img_address)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            # 0-80 行是车头的部分；140-160 行是天空，对于判断方向盘转动没有帮助，故裁掉
            X[example, :, :, :] = cv.resize(img[80:140, 0:320], (shape[0], shape[1])) / 255 - 0.5
            Y[example] = img_steering

            if (example + 1) + offset > len(new_y) - 1:  # 到达原来数据的结尾, 从头开始
                if training:
                    x, y = shuffle(x, y)  # 保证训练时每个 epoch 数据顺序不一样
                    rand_zero_idx = discard_zero_steering(y_train, rate=discard_rate)
                    new_x = np.delete(x, rand_zero_idx, axis=0)
                    new_y = np.delete(y, rand_zero_idx, axis=0)
                else:  # 产生 validation 数据
                    new_x = x
                    new_y = y
                offset = 0

        yield X, Y
        '''
        关于 yield:
        https://www.runoob.com/w3cnote/python-yield-used-analysis.html
        https://blog.csdn.net/mieleizhi0522/article/details/82142856
        '''
        offset += batch_size


if __name__ == '__main__':

    csv_data = []
    with open(data_path + 'driving_log.csv', 'r') as csvfile:
        csv_file = csv.reader(csvfile, delimiter=',')
        for row in csv_file:
            csv_data.append(row)
    csv_data = np.array(csv_data)

    # 判断 csv 中图片数量是否等于文件夹中的图片数量
    imgs_exist = glob.glob(data_path+'IMG/*.jpg')
    assert len(imgs_exist) == len(csv_data) * 3, 'number of images does not match'

    x_ = csv_data[:, 0]  # 只把 center 值取出来
    y_ = csv_data[:, 3].astype(float)
    x_, y_ = shuffle(x_, y_)
    '''
    因为我们不希望网络学习到前后的关联性，所以需要 shuffle
    sklearn.utils.shuffle 此方法把 x_ 和 y_ 一起 shuffle 所以结果还是对应的，如果数据长度不一样会报错
    '''
    X_train, X_val, y_train, y_val = train_test_split(x_, y_, test_size=0.2, random_state=SEED)  # 使用 20% 的数据作为测试数据
    print('X_train.shape', X_train.shape, type(X_train))
    print('X_val.shape', X_val.shape, type(X_val))
    print('y_train.shape', y_train.shape, type(y_train))
    print('y_val.shape', y_val.shape, type(y_val))

    model = get_model()
    # model = load_model('best_model.h5')

    callbacks = [
        ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
        # EarlyStopping(monitor='val_loss', min_delta=0, patience=25, verbose=1, mode='auto')
    ]
    history = model.fit_generator(batch_generator(X_train, y_train),
                                  steps_per_epoch=100,
                                  # 取整 https://blog.csdn.net/weixin_41712499/article/details/85208928
                                  validation_steps=math.ceil(len(y_val) / batch_size),
                                  validation_data=batch_generator(X_val, y_val, training=False),
                                  epochs=60, verbose=1, callbacks=callbacks)

    with open('./trainHistoryDict.p', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.savefig('train_val_loss.jpg')
