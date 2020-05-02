import numpy
import calculation
from scipy.stats import spearmanr
import logging


# Computes the Embedding Coherence Test (ECT) on a bias specification
def embedding_coherence_test(t1, t2, a1, a2):
    logging.info("Eval-Engine: ECT started")
    # Transform vector sets in lists
    attributes_dict = {}
    for word in a1:
        attributes_dict[word] = a1[word]
    for word in a2:
        attributes_dict[word] = a2[word]

    t1_list = calculation.transform_dict_to_list(t1)
    t2_list = calculation.transform_dict_to_list(t2)
    attributes = calculation.transform_dict_to_list(attributes_dict)
    # logging.info("ECT: Vector dictionaries/lists prepared successfully")
    mean_target_vector1 = target_set_mean_vector(t1_list)
    mean_target_vector2 = target_set_mean_vector(t2_list)
    # logging.info("ECT: Target set mean vectors calculated successfully")
    array_sim1 = []
    array_sim2 = []
    for i in range(len(attributes)):
        memory = attributes[i]
        cos_sim1 = calculation.cosine_similarity(mean_target_vector1, memory)
        array_sim1.append(cos_sim1)
        cos_sim2 = calculation.cosine_similarity(mean_target_vector2, memory)
        array_sim2.append(cos_sim2)
    value_array, p_value = spearmanr(array_sim1, array_sim2)
    logging.info("Eval-Engine: ECT-Scores: " + str(value_array) + "; p-value: " + str(p_value))
    return value_array, p_value


# Calculates the mean vector of a target list
def target_set_mean_vector(target_list):
    # Create empty vector with dimension 300 for simpler vector addition
    vector_array = numpy.zeros(300)
    # print("Zeros:" + str(len(vector_array)))
    for i in range(len(target_list)):
        # print(len(target_list[i]))
        vector_array += target_list[i]
    result = (vector_array / len(target_list))
    return result
