from feature_extractor import SIFT
from ir_utils import *
import json
import pickle
import numpy as np

def get_word_id(word_model, word):
    _id = word_model['model'].predict([word])[0]
    if 'id' in word_model:
        return _id + word_model['id']
    return get_word_id(word_model['submodel'][_id], word)

def inverted_index(word_model, inverted_file, words, document_name):
    ids = []
    for word in words:
        ids.append(get_word_id(word_model, word))
    for _id in ids:
        if document_name not in inverted_file[_id]:
            inverted_file[_id][document_name] = 0
        inverted_file[_id][document_name] += 1

def create_sift_inverted_file(dataset='./data/cifar-10/train.json', output='sift_inverted_file.pickle', model_path='words_model.pickle'):
    with open(dataset) as f:
        images = json.load(f)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    count = {'id': 0}
    def assign_model_id(model, count):
        if ('submodel' not in model) or (len(model['submodel']) <= 0):
            model['id'] = count['id']
            count['id'] += model['model'].cluster_centers_.shape[0]
        else:
            for submodel in model['submodel']:
                assign_model_id(submodel, count)
    assign_model_id(model, count)

    extractor = SIFT()

    inverted_file = []
    for i in range(512):
        inverted_file.append(dict())

    for cnt, img in enumerate(images):
        image = read_image_from_config(img)
        kp, des = extractor.extract_full(image)
        if len(kp) == 0:
            continue
        des /= 255.0
        inverted_index(word_model=model, inverted_file=inverted_file, words=des, document_name=img['image_path'])
        if (cnt + 1) % 100 == 0:
            print(cnt + 1)

    with open(output, 'wb') as f:
        pickle.dump(inverted_file, f)

if __name__ == '__main__':
    create_sift_inverted_file()