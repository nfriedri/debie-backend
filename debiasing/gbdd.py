import calculation
import numpy
import logging


def generalized_bias_direction_debiasing(target_set1, target_set2):
    logging.info("GBDD: Debiasing started")
    target1_copy, target2_copy = calculation.create_duplicates(target_set1, target_set2)
    target1_copy2, target2_copy2 = calculation.create_duplicates(target_set1, target_set2)
    target1, target2 = calculation.transform_multiple_dicts_to_lists(target1_copy2, target2_copy2)
    logging.info("GBDD: Vector dictionaries and lists prepared successfully")
    gbdv = calculate_bias_direction_matrix(target1, target2)
    new_target1 = calculate_gbdd(gbdv, target1_copy)
    new_target2 = calculate_gbdd(gbdv, target2_copy)
    logging.info("GBDD: Debiasing finished successfully")
    return new_target1, new_target2


def calculate_bias_direction_matrix(target_list1, target_list2):
    logging.info("GBDD: Calculating bias direction matrix")
    matrix = []
    for i in range(len(target_list1)):
        for j in range(len(target_list2)):
            print(i)
            print(j)
            print()
            array = numpy.array(target_list1[i]) - numpy.array((target_list2[j]))
            matrix.append(array)
    u, s, vh = numpy.linalg.svd(matrix)
    vh_transposed = numpy.transpose(vh)
    gbdv = []
    for i in range(len(vh_transposed)):
        gbdv.append(vh_transposed[i][1])
    logging.info("GBDD: Bias direction matrix calculated successfully")
    return gbdv


def calculate_gbdd(gdv, dict1):
    logging.info("GBDD: Calculating debiased vectors")
    list_words = [word for word in dict1]
    list_vecs = [list(dict1[vec]) for vec in dict1]
    gbdd_vecs = []
    gdv = numpy.array(gdv)
    for i in range(len(list_vecs)):
        value = numpy.array(list_vecs[i]) - (numpy.dot(list_vecs[i], gdv) * gdv)
        gbdd_vecs.append(value)
    logging.info("GBDD: Debiased vectors calculates successfully")
    return create_dict_from_lists(list_words, gbdd_vecs)


def create_dict_from_lists(word_list, vector_list):
    dict1 = {}
    for i in range(len(word_list)):
        dict1[word_list[i]] = vector_list[i]
    return dict1
