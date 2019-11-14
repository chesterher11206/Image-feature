import os
import pickle
import numpy as np
from descriptor.histogram import Histogram
from descriptor.gabor import Gabor
from descriptor.SIFT import SIFT


DB_PATH = 'database'
FEATURE_PATH = 'features'

def read_database():
    database_path = 'database'

    for root, subdirs, files in os.walk(database_path):
        for subdir in subdirs:
            print(subdir)
            subdir_path = os.path.join(root, subdir)
            for subdir_root, subdir_subdirs, subdir_files in os.walk(subdir_path):
                for subdir_file in subdir_files:
                    ext = os.path.splitext(subdir_file)[-1]
                    if ext != '.jpg' and ext != '.jpeg':
                        continue

                    subdir_file_path = os.path.join(subdir_root, subdir_file)
                    file_index = os.path.splitext(subdir_file)[0].split('_')[-1]

                    yield subdir_file_path, subdir, file_index

def extract_feature(descriptor):
    features = dict()

    for image_path, image_category, image_index in read_database():
        if image_category not in features:
            features[image_category] = dict()
        feature = descriptor.describe(image_path)
        features[image_category][image_index] = feature

def save_feature(features, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(features, fp)

def main():
    # COLOR FEATURES
    # descriptor = Histogram([4, 4, 4])
    # descriptor.set_database(DB_PATH)
    # descriptor.extract_features()
    # descriptor.save_features(os.path.join(FEATURE_PATH, 'gch.p'))

    # TEXTURE FEATURES
    # descriptor = Gabor([7, 9, 11, 13, 15, 17], np.arange(0, np.pi, np.pi / 4))
    # descriptor.set_database(DB_PATH)
    # descriptor.extract_features()
    # descriptor.save_features(os.path.join(FEATURE_PATH, 'gabor.p'))

    # LOCAL FEATURES
    descriptor = SIFT()
    descriptor.set_database(DB_PATH)
    descriptor.extract_features(os.path.join(FEATURE_PATH, 'sift.p'))
    # descriptor.save_features(os.path.join(FEATURE_PATH, 'sift.p'))


if __name__ == '__main__':
    main()