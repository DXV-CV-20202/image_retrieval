import numpy as np
import cv2

class FeatureExtractor:
    def __init__(self, *args, **kwargs):
        pass
    
    def extract(self, image, *args, **kwargs):
        raise Exception("extract function must be implemented")

class Random(FeatureExtractor):
    def __init__(self, *args, feature_size=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_size = feature_size

    def extract(self, image, *args, **kwargs):
        return np.random.rand(*self.feature_size)

class HuMoments(FeatureExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def extract(self, image, *args, **kwargs):
        shape = image.shape
        if len(shape) > 2:
            image = np.average(image, axis=tuple(range(2, len(shape))))
        moments = cv2.moments(image)
        hu_moments = cv2.HuMoments(moments)
        hu_moments = np.squeeze(hu_moments)
        log_hu_moments = -1 * np.copysign(1.0, hu_moments) * np.log10(np.abs(hu_moments))
        return log_hu_moments

class SIFT(FeatureExtractor):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.extractor = cv2.SIFT_create()
        self.eps = 1e-7
        self.isRootSIFT = False
    
    def extract(self, image, *args, **kwargs):
        kp, descriptor = self.extractor.detectAndCompute(image, None)
        if self.isRootSIFT == True:
            descriptor /= (descriptor.sum(axis=1, keepdims=True) + self.eps)
            descriptor = np.sqrt(descriptor)
        return kp, descriptor

class HOG(FeatureExtractor):
    def __init__(self, *args, winSize=(32, 32), blockSize=(32, 32), blockStride=(2, 2), cellSize=(16, 16), nbins=9, **kwargs):
        super().__init__(*args, **kwargs)
        self.winSize = winSize # Image size
        self.blockSize = blockSize # multiple of cell size, for histogram normalization
        self.blockStride = blockStride # block overlapping
        self.cellSize = cellSize # each cell has 1 histogram
        self.nbins = nbins # number of directions
        self.extractor = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    
    def extract(self, image, *args, **kwargs):
        image = cv2.resize(image, self.winSize, interpolation = cv2.INTER_AREA)
        return self.extractor.compute(image)