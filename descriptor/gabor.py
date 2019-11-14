import cv2
import numpy as np
import pylab as pl
from .base import Descriptor


class Gabor(Descriptor):
    def __init__(self, scales, rotations):
        self.scales = scales
        self.rotations = rotations
        self.build_filters()

    def describe(self, filename):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered_images = [] #滤波结果
        for kernel in self.filters:        
            filtered_image = cv2.filter2D(image, cv2.CV_8UC3, kernel)
            filtered_images.append(np.asarray(filtered_image))

        feature = []
        for filtered_image in filtered_images:
            mean = np.mean(filtered_image.flatten())
            std = np.std(filtered_image.flatten())
            feature.append(mean)
            feature.append(std)
        feature = np.array(feature)

        return feature

    def build_filters(self):
        filters = []
        for rotation in self.rotations: #gabor方向，0°，45°，90°，135°，共四个
            for scale in self.scales: 
                kernel = cv2.getGaborKernel((scale, scale), 1.0, rotation, np.pi / 2.0, 0.5, 0, ktype=cv2.CV_32F)
                kernel /= 1.5 * kernel.sum()
                filters.append(kernel)
        self.filters = filters
