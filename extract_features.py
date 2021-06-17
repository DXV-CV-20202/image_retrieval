import json
from copy import deepcopy as dcp

import cv2
import pymongo
from multiprocessing import Process
import os
import time

from ir_utils import *

def extract(images, extractors, proc_id, db_config):
    print('Starting process ', proc_id)

    client = pymongo.MongoClient(host=db_config['host'], port=db_config['port'])
    db = client.image_retrieval
    col = db.image_features
    
    for ind, i in enumerate(images):
        image = read_image_from_config(i)
        i['features'] = {}
        for extractor in extractors:
            i['features'][extractor['config']['name']] = extractor['extractor'].extract(image).tolist()
        chunk_size = 500
        if (ind + 1) % chunk_size == 0:
            print("Process ", proc_id, ": ", ind + 1, '/', len(images))
            col.insert_many(images[ind-chunk_size+1:ind+1])
        elif ind + 1 >= len(images):
            print("Process ", proc_id, ": ", ind + 1, '/', len(images))
            col.insert_many(images[ind-len(images)%chunk_size+1:ind+1])
    client.close()

def extract_and_save(dataset='./data/cifar-10/train.json', extractor_config='./config/feature_extractor.json', database_config='./config/database.json'):
    with open(database_config) as f:
        db_config = json.load(f)
    mongodb_config = db_config['mongodb']
    client = pymongo.MongoClient(host=mongodb_config['host'], port=mongodb_config['port'])
    db = client.image_retrieval
    col = db.image_features
    col.delete_many({})
    client.close()

    with open(extractor_config) as f:
        config = json.load(f)
    extractors = []
    for c in config:
        extractors.append(create_extractor(c))
    with open(dataset) as f:
        images = json.load(f)

    processes = []
    num_processes = os.cpu_count()

    chunk_size = int(len(images) / num_processes)
    time_start = time.time()
    for i in range(num_processes):
        start_idx = chunk_size * i
        end_idx = start_idx + chunk_size

        p = Process(target=extract, args=(images[start_idx:end_idx], extractors, i, mongodb_config))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print('Done')
    print('Time:', time.time() - time_start)


if __name__ == '__main__':
    extract_and_save()
