import cv2
import numpy as np

class HuMoments():
    def __init__(self, image):
        self.moment = cv2.moments(image)
        self.humoments = cv2.HuMoments(self.moment)

    def compute(self):
        for i in range(0, 7):
            self.humoments[i] = abs(-np.copysign(1.0, self.humoments[i]) * np.log10(np.abs(self.humoments[i])))
        descriptor = self.humoments.flatten()
        return descriptor
