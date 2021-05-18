import cv2
import numpy as np

class SIFT():
    isRootSift = False
    eps = 1e-7

    def __init__(self):
        self.extractor = cv2.SIFT_create()

    def compute(self, image):
        kp, descriptor = self.extractor.detectAndCompute(image, None)
        if self.isRootSift == True:
            descriptor /= (des.sum(axis=1, keepdims=True) + self.eps)
            descriptor = np.sqrt(des)
        return (kp, descriptor)
