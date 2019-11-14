import os
import cv2
import pickle
import numpy as np


class Descriptor(object):
    def set_database(self, db):
        self.database = db

    def read_database(self):
        database_path = self.database

        for root, subdirs, files in os.walk(database_path):
            for subdir in subdirs:
                subdir_path = os.path.join(root, subdir)
                for subdir_root, subdir_subdirs, subdir_files in os.walk(subdir_path):
                    for subdir_file in subdir_files:
                        ext = os.path.splitext(subdir_file)[-1]
                        if ext not in ('.jpg', '.jpeg'):
                            continue

                        subdir_file_path = os.path.join(subdir_root, subdir_file)
                        file_index = os.path.splitext(subdir_file)[0].split('_')[-1]

                        yield subdir_file_path, subdir, file_index

    def describe(self, filename):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.flatten()

    def extract_features(self):
        if not self.database:
            return

        features = dict()
        for image_path, image_category, image_index in self.read_database():
            if image_category not in features:
                features[image_category] = dict()
            feature = self.describe(image_path)
            features[image_category][image_index] = feature

        self.features = features

    def save_features(self, filename):
        if not self.features:
            return

        with open(filename, 'wb') as fp:
            pickle.dump(self.features, fp)
