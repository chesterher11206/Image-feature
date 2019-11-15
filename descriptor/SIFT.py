import os
import cv2
import numpy as np
import pickle
from sklearn.cluster import KMeans
from .base import Descriptor


FEATURE_PATH = 'features'

class SIFT(Descriptor):
    def __init__(self, vw_num):
        self.vw_num = vw_num

    def extract_features(self):
        if not self.database:
            return

        self.build_sift()
        self.build_codebook()

        self.features = dict()
        for image_category in self.sift_features.keys():
            for image_index in self.sift_features[image_category]:
                if image_category not in self.features:
                    self.features[image_category] = dict()
                feature = self.codebook.predict(self.sift_features[image_category][image_index])
                feature = np.histogram(feature, bins=np.arange(self.vw_num + 1), density=True)
                self.features[image_category][image_index] = feature[0].ravel().astype('float32')

    def build_sift(self):
        self.sift_features = dict()
        for image_path, image_category, image_index in self.read_database():
            if image_category not in self.sift_features:
                self.sift_features[image_category] = dict()
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create()
            kp, des = sift.detectAndCompute(image, None)
            self.sift_features[image_category][image_index] = des

        with open(os.path.join(FEATURE_PATH, 'sift_features.p'), 'wb') as fp:
            pickle.dump(self.sift_features, fp)

    def build_codebook(self):
        sift_features = np.array([], dtype=np.int64).reshape(0, 128)
        for image_category in self.sift_features.keys():
            for image_index in self.sift_features[image_category]:
                sift_features = np.vstack([sift_features, self.sift_features[image_category][image_index]])
        codebook = KMeans(n_clusters=self.vw_num, random_state=0).fit(sift_features)
        self.codebook = codebook

        with open(os.path.join(FEATURE_PATH, 'sift_codebook.p'), 'wb') as fp:
            pickle.dump(self.codebook, fp)
