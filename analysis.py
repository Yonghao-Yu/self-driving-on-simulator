import csv
import glob
import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def angle_over_time(angle_list):
    x = np.arange(0, len(angle_list))
    y = list(angle_list)
    plt.plot(x, y, 'o-r')
    plt.title('Sequence of steering angles')
    plt.xlabel('frame')
    plt.ylabel('steering angle (rad)')
    plt.show()


def freq_distribution(angle_list):
    x = list(angle_list)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    plt.subplots_adjust(wspace=0.5)

    ax1.hist(x, bins=50)
    ax1.set_xlabel('steering angle (rad)')
    ax1.set_ylabel('count')

    ax2.hist(x, bins=50)
    ax2.set_xlabel('steering angle (rad)')
    ax2.set_ylabel('log(count)')
    ax2.set_yscale('log')

    plt.show()


def img_show(img_list):
    while True:
        img_path = random.choice(img_list)
        img = cv.imread(img_path)
        cv.imshow('img', img)
        if cv.waitKey() == ord('q'):
            break
    cv.destroyAllWindows()


def count_num(angle_list):
    total_num = len(angle_list)
    pos_num = len(angle_list[angle_list > 0])
    neg_num = len(angle_list[angle_list < 0])
    zero_num = len(angle_list[angle_list == 0])
    print('positive number:', pos_num/total_num)
    print('negative number:', neg_num / total_num)
    print('zero number:    ', zero_num / total_num)


if __name__ == '__main__':

    data_path = 'data/'
    csv_data = []
    with open(data_path+'driving_log.csv', 'r') as csvfile:
        csv_file = csv.reader(csvfile, delimiter=',')
        for row in csv_file:
            csv_data.append(row)
    csv_data = np.array(csv_data)

    # 判断 csv 中图片数量是否等于文件夹中的图片数量
    imgs_exist = glob.glob(data_path+'IMG/*.jpg')
    assert len(imgs_exist) == len(csv_data)*3, 'number of images does not match'

    steering_angle_list = csv_data[:, 3].astype(float)
    img_list = csv_data[:, 0:3].ravel()

    # 方向盘转动角度随时间的变化
    # angle_over_time(steering_angle_list)

    # 角度分布直方图
    # freq_distribution(steering_angle_list)

    # 查看图片
    # img_show(img_list)

    # 个数统计
    count_num(steering_angle_list)