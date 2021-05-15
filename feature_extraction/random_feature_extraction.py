from .feature_extraction import FeatureExtraction
import numpy as np

class RandomFeatureExtraction(FeatureExtraction):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'feature_size' in kwargs:
            self.output_size = kwargs['feature_size']
        else:
            self.output_size = (16,)

    def extract_feature(self, image, *args, **kwagrgs):
        return np.random.rand(*self.output_size)