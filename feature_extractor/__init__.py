import cv2
import numpy as np
from numpy.core.fromnumeric import squeeze
import torch
import torchvision
import sys
sys.path.append("..")
from image_representation_learning.networks import EmbeddingNet, EmbeddingNetL2, ResNetEmbedding, TripletNet


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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, descriptor = self.extractor.detectAndCompute(image, None)
        if self.isRootSIFT == True:
            descriptor /= (descriptor.sum(axis=1, keepdims=True) + self.eps)
            descriptor = np.sqrt(descriptor)
        return kp, descriptor

class HOG(FeatureExtractor):
    def __init__(self, *args, winSize=(32, 32), blockSize=(32, 32), blockStride=(16, 16), cellSize=(16, 16), nbins=9, **kwargs):
        super().__init__(*args, **kwargs)
        self.winSize = winSize # Image size
        self.blockSize = blockSize # multiple of cell size, for histogram normalization
        self.blockStride = blockStride # block overlapping
        self.cellSize = cellSize # each cell has 1 histogram
        self.nbins = nbins # number of directions
        self.extractor = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    
    def extract(self, image, *args, **kwargs):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, self.winSize, interpolation = cv2.INTER_AREA)
        features = self.extractor.compute(image)
        features = np.squeeze(features)
        return features

class ColorHistogram(FeatureExtractor):
    def __init__(self, *args, nbins = 8, type='RGB', **kwargs):
        super().__init__(*args, **kwargs)
        self.nbins = nbins
        self.type = type
    
    def extract(self, image, *args, **kwargs):
        if self.type == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            histograms  = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            cv2.normalize(histograms, histograms)
            return histograms.flatten()
        elif self.type == 'RGB':
            b, g, r = cv2.split(image)
            rgb_hist = np.zeros((768,1), dtype='uint32')
            b_hist = cv2.calcHist([b], [0], None, [256], [0, 256])
            g_hist = cv2.calcHist([g], [0], None, [256], [0, 256])
            r_hist = cv2.calcHist([r], [0], None, [256], [0, 256])
            rgb_hist = np.array([r_hist, g_hist, b_hist])
            cv2.normalize(rgb_hist, rgb_hist)
            return rgb_hist.flatten()
        else:
            return np.zeros(self.nbins * 3)

class DeepRepresentation(FeatureExtractor):
    def __init__(self, checkpoint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        embedding_net = EmbeddingNet()
        model = TripletNet(embedding_net)
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
        self.model = model
        self.mean = (0.1307,) 
        self.std = (0.3081,)
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])
    
    def extract(self, image, *args, **kwargs):
        image = self.transforms(image)
        image.unsqueeze_(0)
        features = self.model.get_embedding(image)
        features = features[0]
        features = features.tolist()
        features = np.array(features)
        return features

class DeepRepresentationL2(FeatureExtractor):
    def __init__(self, checkpoint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        embedding_net = EmbeddingNetL2()
        model = TripletNet(embedding_net)
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
        self.model = model
        self.mean = (0.1307,) 
        self.std = (0.3081,)
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])
    
    def extract(self, image, *args, **kwargs):
        image = self.transforms(image)
        image.unsqueeze_(0)
        features = self.model.get_embedding(image)
        features = features[0]
        features = features.tolist()
        features = np.array(features)
        return features

class DeepRepresentationResNet(FeatureExtractor):
    def __init__(self, checkpoint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        embedding_net = ResNetEmbedding()
        model = TripletNet(embedding_net)
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
        model.eval()
        self.model = model
        self.mean = (0.1307,) 
        self.std = (0.3081,)
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])
    
    def extract(self, image, *args, **kwargs):
        image = self.transforms(image)
        image.unsqueeze_(0)
        features = self.model.get_embedding(image)
        features = features[0]
        features = features.tolist()
        features = np.array(features)
        return features