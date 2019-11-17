from flask import jsonify
from bias_evaluation import weat, ect, bat, k_means
import logging


def return_bias_evaluation(methods, target1, target2, arg1, arg2):
    logging.info("APP-BE: Forwarding to related definitions")
    if methods is None:
        return return_eval_all(target1, target2, arg1, arg2)
    if methods == 'allBtn':
        return return_eval_all(target1, target2, arg1, arg2)
    if methods == 'ectBtn':
        return return_eval_ect(target1, target2, arg1, arg2)
    if methods == 'batBtn':
        return return_eval_bat(target1, target2, arg1, arg2)
    if methods == 'weatBtn':
        return return_eval_weat(target1, target2, arg1, arg2)
    if methods == 'kmeansBtn':
        return return_eval_kmeans(target1, target2)
    return 400


def return_eval_all(test_vectors1, test_vectors2, arg_vectors1, arg_vectors2):
    logging.info("APP-BE: Starting all evaluations")
    ect_value1, p_value1 = ect.embedding_coherence_test(test_vectors1, test_vectors2, arg_vectors1)
    ect_value2, p_value2 = ect.embedding_coherence_test(test_vectors1, test_vectors2, arg_vectors2)
    # bat_result = bat.biased_analogy_test(test_vectors1, test_vectors2, arg_vectors1, arg_vectors2)
    bat_result = 'Currently not available'
    weat_effect_size, weat_p_value = weat.word_embedding_association_test(test_vectors1, test_vectors2, arg_vectors1,
                                                                          arg_vectors2)
    kmeans = k_means.k_means_clustering(test_vectors1, test_vectors2)
    logging.info("APP-BE: Evaluations finished successfully")
    response = jsonify(ect_value1=ect_value1, p_value1=p_value1, p_value2=p_value2, ect_value2=ect_value2,
                       bat_value=bat_result, weat_effect_size=weat_effect_size, weat_pvalue=weat_p_value,
                       k_means=kmeans)
    logging.info("APP-BE: Results: " + str(response))
    return response


def return_eval_ect(test_vectors1, test_vectors2, arg_vectors1, arg_vectors2):
    logging.info("APP-BE: Starting ECT evaluation")
    ect_value1, p_value1 = ect.embedding_coherence_test(test_vectors1, test_vectors2, arg_vectors1)
    ect_value2, p_value2 = ect.embedding_coherence_test(test_vectors1, test_vectors2, arg_vectors2)
    logging.info("APP-BE: ECT finished successfully")
    response = jsonify(ect_value1=ect_value1, p_value1=p_value1, p_value2=p_value2, ect_value2=ect_value2)
    logging.info("APP-BE: Results: " + str(response))
    return response


def return_eval_bat(test_vectors1, test_vectors2, arg_vectors1, arg_vectors2):
    logging.info("APP-BE: Starting BAT evaluation")
    # bat_result = bat.biased_analogy_test(test_vectors1, test_vectors2, arg_vectors1, arg_vectors2)
    bat_result = 'Currently not available'
    logging.info("APP-BE: BAT finished successfully")
    response = jsonify(bat_value=bat_result)
    logging.info("APP-BE: Results: " + str(response))
    return response


def return_eval_weat(test_vectors1, test_vectors2, arg_vectors1, arg_vectors2):
    logging.info("APP-BE: Starting WEAT evaluation")
    weat_effect_size, weat_p_value = weat.word_embedding_association_test(test_vectors1, test_vectors2, arg_vectors1,
                                                                          arg_vectors2)
    logging.info("APP-BE: WEAT finished successfully")
    response = jsonify(weat_effect_size=weat_effect_size, weat_pvalue=weat_p_value)
    logging.info("APP-BE: Results: " + str(response))
    return response


def return_eval_kmeans(test_vectors1, test_vectors2):
    logging.info("APP-BE: Starting KMeans evaluation")
    kmeans = k_means.k_means_clustering(test_vectors1, test_vectors2)
    logging.info("APP-BE: KMeans finished successfully")
    response = jsonify(k_means=kmeans)
    logging.info("APP-BE: Results: " + str(response))
    return response
