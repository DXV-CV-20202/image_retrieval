import os
import sys
import json
import shutil
import itertools
import random

cifar10 = './data/cifar-10'
coil100 = './data/coil-100'
caltech101 = './data/caltech-101'

# dataset_path la 1 trong 3 bien o tren
# ratio la ti le train:test
def create_dataset(dataset_path='', ratio=1):
    if dataset_path == '':
        sys.exit("dataset_path is empty!")
        
    # Tao folder test va train
    train_path = dataset_path + '/train'
    test_path = dataset_path + '/test'
    if os.path.isdir(train_path) == True:
        shutil.rmtree(train_path)

    if os.path.isdir(test_path) == True:
        shutil.rmtree(test_path)

    # Lay danh sach cac class
    class_name = os.listdir(dataset_path)
    print(len(class_name))

    print("Folder train of {} is created !".format(dataset_path))
    os.mkdir(train_path)
    print("Folder test of {} is created !".format(dataset_path))
    os.mkdir(test_path)

    # Voi moi class tao folder trong test va train
    for cn in class_name:
        cn_train_path = train_path + '/' + cn
        cn_test_path = test_path + '/' + cn
        if os.path.isdir(cn_train_path) != True:
            os.mkdir(cn_train_path)
        if os.path.isdir(cn_test_path) != True:
            os.mkdir(cn_test_path)

    # Shuffle moi class voi train:test = ratio r di chuyen vao train voi test
    for cn in class_name:
        list_image = []
        cn_path = dataset_path + '/' + cn
        for f in os.listdir(cn_path):
            if f.endswith('.png') or f.endswith('.jpg'):
                list_image.append(f)
        random.shuffle(list_image)
        list_image_train = list_image[:int(len(list_image) * ratio)]
        list_image_test = list_image[int(len(list_image) * ratio):]
        
        for i in list_image_train:
            i_train_path = cn_path + '/' + i
            i_train_folder = train_path + '/' + cn
            shutil.move(i_train_path, i_train_folder)

        for i in list_image_test:
            i_test_path = cn_path + '/' + i
            i_test_folder = test_path + '/' + cn
            shutil.move(i_test_path, i_test_folder)

        if len(os.listdir(cn_path)) == 0:
            os.rmdir(cn_path)


if __name__ == '__main__':
    create_dataset(dataset_path=caltech101 , ratio=0.8)
