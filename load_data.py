import numpy as np
import os
import cv2
import random
import tensorflow as tf

def load_image(path_dataset):
    dataset = []
    for name in path_dataset:
        img = cv2.imread(name)
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=-1)
        dataset.append(img)
    return (np.array(dataset).astype('float32')) /127.5 - 1

def load_ck(path = '/home/pomelo96/Desktop/datasets/classifier_alignment_CK', train=True):
    if train:
        path = path + '/' + 'train'
    else:
        path = path + '/' + 'test'
    natural_img_path_set = []
    expression_img_path_set = []

    natural_path = path + '/' + 'Natural image'
    natural_id_path = os.listdir(natural_path)
    natural_id_path.sort()
    for id in natural_id_path:
        sub_natural_path = natural_path + '/' + id
        sub_natural_path = sub_natural_path + '/' + os.listdir(sub_natural_path)[0]
        natural_image_path = os.listdir(sub_natural_path)
        for img_path in natural_image_path:
            natural_img_path_set.append(sub_natural_path + '/' + img_path)

    expression_path = path + '/' + 'Expression image'
    expression_id_path = os.listdir(expression_path)
    expression_id_path.sort()
    for id in expression_id_path:
        sub_expression_path = expression_path + '/' + id
        sub_expression_path = sub_expression_path + '/' + os.listdir(sub_expression_path)[0]
        expression_image_path = os.listdir(sub_expression_path)
        for img_path in expression_image_path:
            expression_img_path_set.append(sub_expression_path + '/' + img_path)

    return natural_img_path_set, expression_img_path_set

def get_batch_data(data,batch_idx,batch_size):
    range_min = batch_idx * batch_size
    range_max = (batch_idx + 1) * batch_size
    if range_max > len(data):
        range_max = len(data)
    index = list(range(range_min, range_max))
    temp_data = [data[idx] for idx in index]
    return temp_data

def build_pretrain_data(natural_roots, expression_roots):
    total_roots = []
    [[total_roots.append(i) for i in j] for j in [natural_roots, expression_roots]]  #black tech dont ask
    natural_label = build_ck_label(natural_roots, label_type='natural')
    expression_label = build_ck_label(expression_roots, label_type='expression')
    label = []
    [[label.append(i) for i in j] for j in [natural_label, expression_label]]
    temp = list(zip(total_roots, label))
    random.shuffle(temp)
    total_roots, label = zip(*temp)
    return total_roots, label   #roots with label 0 or 1

def build_train_data(natural_roots):
    natural_label = build_ck_label(natural_roots, 'natural')
    expression_label = build_ck_label(natural_roots, 'expression')
    temp = list(zip(natural_roots, natural_label, expression_label))
    random.shuffle(temp)
    natural_roots, natural_label, expression_label = zip(*temp)

    return natural_roots, natural_label, expression_label

def build_ck_label(data, label_type):
    if label_type == 'natural':
        label = [0] * (len(data))
    elif label_type == 'expression':
        label = [1] * (len(data))

    return label

def build_training_label(label, total_class=2, img_shape=(128,128)):

    training_label = np.zeros((len(label), img_shape[0], img_shape[1], total_class))    #build one-hot-vector
    for i in range(len(label)):
        training_label[i,:,:,label[i]] = training_label[i,:,:,label[i]] + 1        #ith and label[i] channel be 1

    return tf.cast(training_label, dtype='float32')

def load_ck_by_id(i, train):
    if train:
        path = '/home/pomelo96/Desktop/datasets/classifier_alignment_CK/train/Expression image'
    else:
        path = '/home/pomelo96/Desktop/datasets/classifier_alignment_CK/test/Expression image'
    path_dataset = []
    id_file = os.listdir(path)
    id_file.sort()
    id_ = id_file[i]
    id_img_file = path + '/' + id_
    id_img_file = id_img_file + '/' + os.listdir(id_img_file)[0]
    id_img_name = os.listdir(id_img_file)
    for name in id_img_name:
        img_path = id_img_file + '/' + name
        path_dataset.append(img_path)
    img_dataset = load_image(path_dataset)
    return img_dataset, path_dataset, id_


if __name__ == '__main__':
    natural_roots_train, expression_roots_train = load_ck(train=True)
    natural_roots_test, expression_roots_test = load_ck(train=False)
    total_roots_train, label_train = build_pretrain_data(natural_roots_train,
                                                                   expression_roots_train)
    total_roots_test, label_test = build_pretrain_data(natural_roots_test, expression_roots_test)
    print(len(total_roots_train), len(label_train))
    for i in range(10):
        print(total_roots_train[i], label_train[i])
    natural_cnt = 0
    expression_cnt = 0
    for i in range(len(label_train)):
        if label_train[i] == 0:
            natural_cnt += 1
        elif label_train[i] == 1:
            expression_cnt += 1
    print(natural_cnt, expression_cnt)

    natural_cnt = 0
    expression_cnt = 0
    for i in range(len(label_test)):
        if label_test[i] == 0:
            natural_cnt += 1
        elif label_test[i] == 1:
            expression_cnt += 1
    print(natural_cnt, expression_cnt)
