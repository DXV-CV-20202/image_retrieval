from copy import deepcopy
from matplotlib import pyplot as plt
import itertools
import cv2
import numpy as np

def create_extractor(config):
    components = config['extractor'].split('.')
    _module = __import__(components[0])
    for component in components[1:]:
        _module = getattr(_module, component)
    extractor = {}
    extractor['config'] = deepcopy(config)
    extractor['extractor'] = _module(**config['parameters'])
    return extractor

# Them padding cho anh caltech101 de resize ve (128,128)
def add_padding(image):
    height = image.shape[0]
    width = image.shape[1]
    p = abs(width - height)//2
    if width > height:
        padding = cv2.copyMakeBorder(image, p,p,0,0, cv2.BORDER_CONSTANT, value=0)
    else:
        padding = cv2.copyMakeBorder(image, 0,0,p,p, cv2.BORDER_CONSTANT, value=0)
    return padding

# Crop lay doi tuong trong anh
def cropImage(image, x, y, w, h):
    if np.ndim(image) == 3:
        crop = image[y:y+h, x:x+w, :]
    else:
        crop = image[y:y+h, x:x+w]
    return crop

# Tu dong tinh nguong bien anh
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

# Thay doi ham nay de xu ly rieng tung dataset, tung extractor
def read_image_from_config(config, dataset=None, extractor=None):
    image_path = config['image_path']

    # Voi dataset la coil-100
    if dataset == 'coil-100':
        if extractor == 'HuMoments':
            image = cv2.imread(image_path, 0)
            image = cv2.GaussianBlur(image, (5,5), 0)
            return image
        elif extractor == 'HOG':
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            blur = cv2.bilateralFilter(image, 9, 75, 75)
            return image
        elif extractor == 'HOG_HSV':
            image = cv2.imread(image_path)
            hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue, _, _ = cv2.split(hsv_img)
            blur = cv2.bilateralFilter(hue, 9, 75, 75)
            return image
        elif extractor == 'SIFT':
            image = cv2.imread(image_path, 0)
            return image
        elif extractor == 'ColorHistogram':
            return cv2.imread(image_path)
        else:
            return cv2.imread(image_path, 0)

    # Voi dataset la caltech-101
    elif dataset == 'caltech-101':
        if extractor == 'HuMoments':
            image = cv2.imread(image_path, 0)
            blur = cv2.GaussianBlur(image, (5,5), 0)
            return blur
        elif extractor == 'HOG':
            image = cv2.imread(image_path, 0)
            blur = cv2.bilateralFilter(image,9,75,75)
            padding = add_padding(blur)
            resize = cv2.resize(padding, (128, 128), interpolation=cv2.INTER_AREA)
            return resize
        elif extractor == 'HOG_HSV':
            image = cv2.imread(image_path)
            hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue, _, _ = cv2.split(hsv_img)
            blur = cv2.bilateralFilter(hue, 9, 75, 75)
            return blur
        elif extractor == 'SIFT':
            return cv2.imread(image_path, 0)
        else:
            return cv2.imread(image_path)
    elif dataset == 'cifar-10':
        if extractor == 'HOG':
            image = cv2.imread(image_path, 0)
        else:
            return cv2.imread(image_path, 0)
    else:
        return cv2.imread(image_path, 0)


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