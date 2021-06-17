import json
import time

import cv2
import numpy as np
import pymongo
from scipy.spatial import KDTree

from ir_utils import *


def main(
    db_cfg_path='./config/database.json',
    connect_name='mongodb',
    extr_cfg_path='./config/feature_extractor.json',
    list_features={'HuMoments'},
    testset_path = './data/cifar-10/test.json'
    ):

    with open(db_cfg_path) as f:
        db_config = json.load(f)
    db_config = db_config[connect_name]
    del db_config['name']
    client = pymongo.MongoClient(**db_config)
    db = client.image_retrieval
    collection = db.image_features
    collection = collection.find()
    # collection = list(collection)

    with open(extr_cfg_path) as f:
        extractors_cfg = json.load(f)
    extractors = []
    for cfg in extractors_cfg:
        if cfg['name'] in list_features:
            extractors.append(create_extractor(cfg)['extractor'])

    class EuclideanDistance:
        def __init__(self) -> None:
            pass
        def calculate_distance(self, x, y):
            return np.linalg.norm(x-y)
    metric = EuclideanDistance()

    matcher = ExhaustiveMatcher(features_name=list_features, extractors=extractors, collection=collection, metric=metric)

    with open(testset_path) as f:
        testset_des = json.load(f)
    
    list_n_top = [1, 3, 5, 10]
    count = dict()
    total = dict()
    for n_top in list_n_top:
        count[n_top] = total[n_top] = 0
    sample_count = 0

    start_time = time.time()
    for des in testset_des:
        image = cv2.imread(des['image_path'])
        res = matcher.match(image)

        image_class_name = des['class_name']

        for n_top in list_n_top:
            records_class_name = [r[0]['image_path'].split('/')[-2] for r in res[:n_top]]
            for record_class_name in records_class_name:
                if image_class_name == record_class_name:
                    count[n_top] += 1
            total[n_top] += len(records_class_name)

        sample_count += 1
        if sample_count % 10 == 0:
            print(sample_count)
        # if sample_count == 100:
        #     break

    print('Time:', time.time() - start_time)
    for n_top in list_n_top:
        print('Top %d accuracy:' % (n_top,), str(count[n_top] * 100 / total[n_top]) + '%')

class Matcher:
    def __init__(self, *args, features_name=[], extractors=[], collection=None, metric=None, **kwargs):
        self.features_name = features_name
        self.extractors = extractors
        self.collection = collection
        self.metric = metric

    def get_features(self, image):
        features = [extractor.extract(image) for extractor in self.extractors]
        features = np.concatenate(features)
        return features

    def get_record_features(self, record):
        features = [record['features'][ft] for ft in self.features_name]
        features = np.concatenate(features)
        return features
    
    def match(self, image, *args, ntop=10, **kwargs):
        pass

class ExhaustiveMatcher(Matcher):
    def __init__(self, *args, features_name=[], extractors=[], collection=None, metric=None, **kwargs):
        super().__init__(*args, features_name=features_name, extractors=extractors, collection=collection, metric=metric, **kwargs)
    
    def match(self, image, *args, ntop=10, **kwargs):
        features = self.get_features(image)
        res = []
        for record in self.collection:
            record_features = self.get_record_features(record)
            distance = self.metric.calculate_distance(features, record_features)
            res.append((record, distance))
            idx = len(res) - 1
            while(idx > 0) and (distance < res[idx-1][1]):
                res[idx], res[idx-1] = res[idx-1], res[idx]
                idx -= 1
            if len(res) > ntop:
                res.pop(-1)
        return res

class KDTreeMatcher(Matcher):
    def __init__(self, *args, features_name=[], extractors=[], collection=None, metric=None, **kwargs):
        super().__init__(*args, features_name=features_name, extractors=extractors, collection=collection, metric=metric **kwargs)
        self.kd_tree = KDTree([record['features']['SIFT'] for record in collection])

    def match(self, image, *args, ntop=10, **kwargs):
        features = self.get_features(image)
        dd, ii = self.kd_tree.query([features], k=ntop)
        dd = dd[0]
        ii = ii[0]
        res = [(self.collection[ii[i]], dd[i]) for i in range(len(ii))]
        return res

if __name__ == '__main__':
    main()
