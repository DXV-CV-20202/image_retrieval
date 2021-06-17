import json
from copy import deepcopy as dcp

import cv2
import pymongo

from ir_utils import *


def extract_and_save(dataset='./data/cifar-10/train.json', extractor_config='./config/feature_extractor.json', database_config='./config/database.json'):
    with open(database_config) as f:
        db_config = json.load(f)
    mongodb_config = db_config['mongodb']
    client = pymongo.MongoClient(host=mongodb_config['host'], port=mongodb_config['port'])
    db = client.image_retrieval
    col = db.image_features
    with open(extractor_config) as f:
        config = json.load(f)
    extractors = []
    for c in config:
        extractors.append(create_extractor(c))
    with open(dataset) as f:
        images = json.load(f)
    for ind, i in enumerate(images):
        image = read_image_from_config(i)
        i['features'] = {}
        for extractor in extractors:
            i['features'][extractor['config']['name']] = extractor['extractor'].extract(image).tolist()
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
            x = col.bulk_write(requests)
            # x = col.insert_many(images[ind-99:ind+1])


if __name__ == '__main__':
    extract_and_save()
