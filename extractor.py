import os
import pickle
import numpy as np
from descriptor.gch import GCH
from descriptor.rch import RCH
from descriptor.gabor import Gabor
from descriptor.SIFT import SIFT


DB_PATH = 'database'
FEATURE_PATH = 'features'

def main():
    # COLOR FEATURES
    # GCH: Global Color Histogram
    descriptor = GCH([12, 12, 12])  # each channel quantized into 12 part
    descriptor.set_database(DB_PATH)
    descriptor.extract_features()
    descriptor.save_features(os.path.join(FEATURE_PATH, 'gch.p'))

    # RCH: Regional Color Histogram
    descriptor = RCH([12, 12, 12], 4) # each channel quantize into 12 part, each image divided into 4*4 subimage
    descriptor.set_database(DB_PATH)
    descriptor.extract_features()
    descriptor.save_features(os.path.join(FEATURE_PATH, 'rch.p'))

    # TEXTURE FEATURES
    descriptor = Gabor([7, 9, 11, 13, 15, 17], np.arange(0, np.pi, np.pi / 4))
    descriptor.set_database(DB_PATH)
    descriptor.extract_features()
    descriptor.save_features(os.path.join(FEATURE_PATH, 'gabor.p'))

    # LOCAL FEATURES
    descriptor = SIFT(100) # cluster SIFT to 100 visual words by KMeans
    descriptor.set_database(DB_PATH)
    descriptor.extract_features()
    descriptor.save_features(os.path.join(FEATURE_PATH, 'sift.p'))


if __name__ == '__main__':
    main()