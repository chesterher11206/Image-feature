import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

ASC = 'ascend'
DESC = 'descend'

FEATURE_PATH = 'features'

COLOR_FEATURE_FILE = 'gch.p'
HISTCMP_METHODS = {
	'Correlation': (cv2.HISTCMP_CORREL, DESC),
	'Chi-Squared': (cv2.HISTCMP_CHISQR, ASC),
	'Intersection': (cv2.HISTCMP_INTERSECT, DESC),
	'Hellinger': (cv2.HISTCMP_BHATTACHARYYA, ASC),
}

TEXTURE_FEATURE_FILE = 'gabor.p'


def leave_one_out(features, similarity_func, orderby=ASC):
    map_per_category = dict()
    map_across_dataset = 0

    for category in features.keys():
        map_per_category[category] = 0
        for index in features[category].keys():
            ap = get_ap(category, index, features, similarity_func, orderby)
            map_per_category[category] += ap
        map_per_category[category] = map_per_category[category] / len(features[category])
        map_across_dataset += map_per_category[category]

    map_across_dataset = map_across_dataset / len(features)
    map_rank = sorted([(v, k) for (k, v) in map_per_category.items()], reverse=True)

    return map_across_dataset, map_rank

def get_ap(target_category, target_index, features, similarity_func, orderby):
    dist_rank = []
    result_rank = []
    query = features[target_category][target_index]
    for category in features.keys():
        for index in features[category].keys():
            if category == target_category and index == target_index:
                continue
            dist = similarity_func(query, features[category][index])

            insert_index = len(dist_rank)
            for i, d in enumerate(dist_rank):
                if dist <= d:
                    insert_index = i
                    break

            dist_rank.insert(insert_index, dist)
            result_rank.insert(insert_index, (category, index))

    if orderby == DESC:
        dist_rank.reverse()
        result_rank.reverse()

    ap = rank2ap(result_rank, target_category)

    return ap

def rank2ap(result_rank, target_category):
    retrieve_num = 0
    retrieve_precision = []
    for rank, (category, index) in enumerate(result_rank):
        if category == target_category:
            retrieve_num += 1
            retrieve_precision.append(retrieve_num / (rank + 1))

    return np.mean(retrieve_precision)

def main():
    # COLOR FEATURES:
    # feature_file = COLOR_FEATURE_FILE
    # method, orderby = HISTCMP_METHODS['Intersection']
    # similarity_func = lambda query, feature: cv2.compareHist(query, feature, method)

    # with open(feature_file, 'rb') as fp:
    #     features = pickle.load(fp)
    # map_across_dataset, map_rank = leave_one_out(features, similarity_func, orderby)
    # print(map_across_dataset)
    # print(map_rank[0], map_rank[1])

    # TEXTURE FEATURES:
    feature_file = TEXTURE_FEATURE_FILE
    similarity_func = lambda query, feature: euclidean(query, feature)

    with open(os.path.join(FEATURE_PATH, feature_file), 'rb') as fp:
        features = pickle.load(fp)
    map_across_dataset, map_rank = leave_one_out(features, similarity_func, ASC)
    print(map_across_dataset)
    print(map_rank[0], map_rank[1])

    # for key in data.keys():
    #     k = plt.bar(np.arange(64), data[key]['1'], .5)
    #     plt.grid(True)
    #     plt.show()


if __name__ == '__main__':
    main()
