import os
import sys
import json
import shutil
import itertools
import cv2

cifar10 = './data/cifar-10'
coil100 = './data/coil-100'
caltech101 = './data/caltech-101'

def create_image_description(dataset=None, subdataset=None, image_path=None, size=None, class_name=None):
    return {
        'dataset': dataset,
        'subdataset': subdataset,
        'image_path': image_path,
        'size': size,
        'class_name': class_name,
        'image_name': '_'.join([dataset, subdataset, class_name, image_path.split('/')[-1]])
    }

def create_dataset_description(dataset_path=''):
    if dataset_path == '':
        sys.exit("dataset_path is empty!")

    #Tao file json
    train_description_path = dataset_path + '/train.json'
    test_description_path = dataset_path + '/test.json'

    if not os.path.isfile(train_description_path):
        with open(train_description_path, 'w'): pass
    else:
        os.remove(train_description_path)
        with open(train_description_path, 'w'): pass

    if not os.path.isfile(test_description_path):
        with open(test_description_path, 'w'): pass
    else:
        os.remove(test_description_path)
        with open(test_description_path, 'w'): pass

    # Lay cac label vao danh sach classes
    classes = os.listdir(dataset_path + '/train')

    train = list()
    test = list()
    for c in classes:
        train_path = '/'.join([dataset_path, 'train', c])
        test_path = '/'.join([dataset_path, 'test', c])
        train_images = os.listdir(train_path)
        test_images = os.listdir(test_path)
        tmp_train = [0] * len(train_images)
        tmp_test = [0] * len(test_images)
        for i, image in enumerate(train_images):
            image_path = '/'.join([train_path, image])
            img = cv2.imread(image_path)
            size = img.shape
            tmp_train[i] = create_image_description(dataset=dataset_path.split('/')[-1], subdataset='train', image_path=image_path, size=size, class_name=c)
        for i, image in enumerate(test_images):
            image_path = '/'.join([test_path, image])
            img = cv2.imread(image_path)
            size = img.shape
            tmp_test[i] = create_image_description(dataset=dataset_path.split('/')[-1], subdataset='test', image_path=image_path, size=size, class_name=c)
        train.append(tmp_train)
        test.append(tmp_test)
    train = list(itertools.chain(*train))
    test = list(itertools.chain(*test))
    with open(train_description_path, 'w') as f:
        json.dump(train, f, indent=4)
    with open(test_description_path, 'w') as f:
        json.dump(test, f, indent=4)


if __name__ == '__main__':
    create_dataset_description(dataset_path=caltech101)
