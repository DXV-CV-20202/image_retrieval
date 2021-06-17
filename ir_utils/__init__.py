from copy import deepcopy

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

def read_image_from_config(config):
    return cv2.imread(config['image_path'])
