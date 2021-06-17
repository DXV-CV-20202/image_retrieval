import os
import json
import itertools

def create_cifar_dataset_description(dataset_path='./data/cifar-10', train_description_path='./data/cifar-10/train_small.json'):

    def create_image_description(dataset='cifar-10', subdataset=None, image_path=None, size=None, class_name=None):
        return {
            'dataset': dataset,
            'subdataset': subdataset,
            'image_path': image_path,
            'size': size,
            'class_name': class_name,
            'image_name': '_'.join([dataset, subdataset, class_name, image_path.split('/')[-1]])
        }

    classes = [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'
    ]
    train = list()
    size=(32, 32, 3)
    for c in classes:
        train_path = '/'.join([dataset_path, 'train', c])
        print(train_path)
        train_images = os.listdir(train_path)[:4000]
        tmp_train = [0] * len(train_images)
        for i, image in enumerate(train_images):
            image_path = '/'.join([train_path, image])
            tmp_train[i] = create_image_description(subdataset='train', image_path=image_path, size=size, class_name=c)
        train.append(tmp_train)
    train = list(itertools.chain(*train))
    with open(train_description_path, 'w') as f:
        json.dump(train, f, indent=4)


if __name__ == '__main__':
    create_cifar_dataset_description()