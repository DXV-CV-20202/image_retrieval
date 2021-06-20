import json
from copy import deepcopy as dcp

import pymongo

from ir_utils import *

import torch

from feature_extractor import DeepRepresentation


def extract_and_save(dataset='./data/cifar-10/train.json', checkpoint='./model/checkpoint.ckp', database_config='./config/database.json'):
    with open(database_config) as f:
        db_config = json.load(f)
    mongodb_config = db_config['mongodb']
    client = pymongo.MongoClient(host=mongodb_config['host'], port=mongodb_config['port'])
    db = client.image_retrieval
    col = db.image_features

    extractor = DeepRepresentation(checkpoint)

    with torch.no_grad():
        with open(dataset) as f:
            images = json.load(f)
        for ind, i in enumerate(images):
            image = read_image_from_config(i)
            i['features'] = {}
            i['features']['deep_representation'] = extractor.extract(image).tolist()
            if (ind + 1) % 100 == 0:
                print(ind + 1, '/', len(images))
                requests = []
                for img in images[ind-99:ind+1]:
                    _set = dict()
                    for ft in img['features']:
                        _set['features.'+ft] = img['features'][ft]
                    requests.append({
                        'filter': {
                            'image_path': img['image_path']
                        },
                        'update': {
                            '$set': _set
                        }
                    })
                requests = [pymongo.UpdateOne(r['filter'], r['update'], upsert=True) for r in requests]
                col.bulk_write(requests)
                # break


if __name__ == '__main__':
    extract_and_save()
