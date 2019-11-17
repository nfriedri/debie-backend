import numpy
from sklearn.cluster import KMeans
import calculation
import random
import logging


def k_means_clustering(target_set1, target_set2, accuracy=50):
    logging.info("KMeans: Calculation started with " + str(accuracy) + " iterations")
    target1, target2 = calculation.create_duplicates(target_set1, target_set2)
    target1 = calculation.transform_dict_to_list(target1)
    target2 = calculation.transform_dict_to_list(target2)
    vector_list = target1 + target2
    logging.info("KMeans: Vector dictionaries and lists prepared successfully")

    gold_standard1 = [1] * len(target1) + [0] * len(target2)
    gold_standard2 = [2] * len(target1) + [0] * len(target2)
    cluster = list(zip(vector_list, gold_standard1, gold_standard2))
    scores = []
    logging.info("KMeans: Cluster & Gold Standards created")
    for i in range(accuracy):
        random.shuffle(cluster)
        k_means = KMeans(n_clusters=2, random_state=0, init='k-means++').fit(numpy.array([x[0] for x in cluster]))
        labels = k_means.labels_
        accuracy1 = len([i for i in range(len(labels)) if labels[i] == cluster[i][1]]) / len(labels)
        accuracy2 = len([i for i in range(len(labels)) if labels[i] == cluster[i][2]]) / len(labels)
        scores.append(max(accuracy1, accuracy2))

    result = sum(scores) / len(scores)
    logging.info("KMeans: Finished calculation")
    logging.info("KMeans: Results: " + str(result))
    return result
