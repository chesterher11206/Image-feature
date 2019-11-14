import cv2
import gc
import numpy as np
from sklearn.decomposition import SparseCoder
from .base import Descriptor


class SIFT(Descriptor):
    def extract_features(self):
        if not self.database:
            return

        print('build sift')
        self.build_sift()
        print('build codebook')
        self.build_codebook(['omp', None, 15])

        print('describe')
        del self.sift_features
        gc.collect()
        self.features = dict()
        for image_path, image_category, image_index in self.read_database():
            if image_category not in self.features:
                self.features[image_category] = dict()
            feature = self.describe(image_path)
            self.features[image_category][image_index] = feature

    def describe(self, filename):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(image, None)
        feature = self.codebook.transform(des)
        feature = np.sum(feature, axis=0)
        feature = cv2.normalize(feature, dst=feature.shape)

        return feature

    def build_sift(self):
        self.sift_features = np.array([], dtype=np.int64).reshape(0, 128)
        for image_path, image_category, image_index in self.read_database():
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create()
            kp, des = sift.detectAndCompute(image, None)
            self.sift_features = np.vstack([self.sift_features, des])

        for x in locals().keys():
            del locals()[x]
            gc.collect()

    def build_codebook(self, param):
        algo, alpha, n_nonzero = param
        codebook = SparseCoder(dictionary=self.sift_features, transform_n_nonzero_coefs=n_nonzero,
                            transform_alpha=alpha, transform_algorithm=algo)
        self.codebook = codebook

        for x in locals().keys():
            del locals()[x]
            gc.collect()
