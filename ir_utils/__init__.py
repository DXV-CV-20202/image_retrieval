from copy import deepcopy
from matplotlib import pyplot as plt
import itertools
import numpy as np

import cv2


def create_extractor(config):
    components = config['extractor'].split('.')
    _module = __import__(components[0])
    for component in components[1:]:
        _module = getattr(_module, component)
    extractor = {}
    extractor['config'] = deepcopy(config)
    extractor['extractor'] = _module(**config['parameters'])
    return extractor


def read_image_from_config(config, dataset=None, extractor=None):
    image_path = config['image_path']
    image = cv2.imread(image_path)
    image = cv2.GaussianBlur(image, (5,5), 0)
    return image

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues, output_file='dataset_extractor'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm * 100
        print("\nNormalized confusion matrix")
    else:
        print('\nConfusion matrix, without normalization')
    print(cm)
    print()
    plt.figure(1, figsize=(30, 30))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.0f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_file, bbox_inches='tight', dpi=500)
    plt.show()
