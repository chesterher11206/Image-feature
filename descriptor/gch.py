import cv2
from .base import Descriptor


class GCH(Descriptor):
    def __init__(self, bins):
        self.bins = bins

    def describe(self, filename):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist(images=[image], channels=[0, 1, 2], mask=None,
                            histSize=self.bins, ranges=[0, 256] * 3)
        hist = cv2.normalize(hist, dst=hist.shape)
        feature = hist.flatten()

        return feature