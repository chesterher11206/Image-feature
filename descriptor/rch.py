import cv2
import numpy as np
from .base import Descriptor


class RCH(Descriptor):
    def __init__(self, bins, region):
        self.bins = bins
        self.region = region

    def describe(self, filename):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        feature = []
        for i in np.array_split(image, self.region):
            for j in np.array_split(i, self.region, axis=1):
                hist = cv2.calcHist(images=[np.array(j)], channels=[0, 1, 2], mask=None,
                                histSize=self.bins, ranges=[0, 256] * 3)
                hist = cv2.normalize(hist, dst=hist.shape)
                feature = np.concatenate((feature, hist.flatten()))
        feature = feature.ravel().astype('float32')

        return feature