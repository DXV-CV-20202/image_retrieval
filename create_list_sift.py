import json
import pickle

from feature_extractor import SIFT
from ir_utils import *


def create_list_sift(dataset='./data/cifar-10/train.json', output='list_sift.pickle'):

    with open(dataset) as f:
        images = json.load(f)

    extractor = SIFT()

    list_sift = []

    for img in images:
        image = read_image_from_config(img)
        kp, des = extractor.extract_full(image)
        if len(kp) > 0:
            list_sift += des.tolist()

    print(len(list_sift))
    
    with open(output, 'wb') as f:
        pickle.dump(list_sift, f)

if __name__ == '__main__':
    create_list_sift()