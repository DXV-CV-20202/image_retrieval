import cv2
import numpy as np
from numpy.core.fromnumeric import squeeze


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
        self.size = 1024

    def extract(self, image, *args, **kwargs):
        kp, descriptor = self.extract_full(image, *args, **kwargs)
        kp_des = [(kp[i], descriptor[i]) for i in range(len(kp))]
        kp_des.sort(key=lambda x: x[0].response, reverse=True)
        if len(kp_des) > 0:
            features = np.concatenate([d[1] for d in kp_des])
            if features.shape[0] < 1024:
                features = np.concatenate([features, np.zeros(1024 - features.shape[0])])
        else:
            features = np.zeros(1024)
        return features[:1024]
    
    def extract_full(self, image, *args, **kwargs):
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
        features = self.extractor.compute(image)
        features = np.squeeze(features)
        return features

class ColorHistogram(FeatureExtractor):
    def __init__(self, *args, nbins = 8, type='', **kwargs):
        super().__init__(*args, **kwargs)
        self.nbins = nbins
        self.type = type
    
    def extract(self, image, *args, **kwargs):
        type = 'HSV'
        if type == 'HSV':
            # convert the image from BGR to HSV  format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # compute the color histograms
            histograms  = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            # normalize the histograms
            cv2.normalize(histograms, histograms)
            # return the histograms
            return histograms.flatten()
        elif type == 'RGB':
            b, g, r = cv2.split(image)
            # compute the color histograms
            rgb_hist = np.zeros((768,1), dtype='uint32')
            b_hist = cv2.calcHist([b], [0], None, [256], [0, 256])
            g_hist = cv2.calcHist([g], [0], None, [256], [0, 256])
            r_hist = cv2.calcHist([r], [0], None, [256], [0, 256])
            rgb_hist = np.array([r_hist, g_hist, b_hist])
            # normalize the histograms
            cv2.normalize(rgb_hist, rgb_hist)
            # return the histograms
            return rgb_hist.flatten()
        return np.zeros(self.nbins * 3)
