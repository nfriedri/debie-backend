from flask import jsonify
from bias_evaluation import weat, ect, bat, k_means
import logging


def return_bias_evaluation(methods, target1, target2, arg1, arg2):
    logging.info("APP-BE: Forwarding to related definitions")
    if methods is None:
        return return_eval_all(target1, target2, arg1, arg2)
    if methods == 'all':
        return return_eval_all(target1, target2, arg1, arg2)
    if methods == 'ect':
        return return_eval_ect(target1, target2, arg1, arg2)
    if methods == 'bat':
        return return_eval_bat(target1, target2, arg1, arg2)
    if methods == 'weat':
        return return_eval_weat(target1, target2, arg1, arg2)
    if methods == 'kmeans':
        return return_eval_kmeans(target1, target2)
    return 400


def return_eval_all(test_vectors1, test_vectors2, arg_vectors1, arg_vectors2):
    logging.info("APP-BE: Starting all evaluations")
    try:
        ect_value1, p_value1 = ect.embedding_coherence_test(test_vectors1, test_vectors2, arg_vectors1)
        ect_value2, p_value2 = ect.embedding_coherence_test(test_vectors1, test_vectors2, arg_vectors2)
        # bat_result = bat.biased_analogy_test(test_vectors1, test_vectors2, arg_vectors1, arg_vectors2)
        bat_result = 'Currently not available'
        weat_effect_size, weat_p_value = weat.word_embedding_association_test(test_vectors1, test_vectors2, arg_vectors1,
                                                                              arg_vectors2)
        kmeans = k_means.k_means_clustering(test_vectors1, test_vectors2)
        logging.info("APP-BE: Evaluations finished successfully")
    except RuntimeWarning as rw:
        print(rw)

    if ect_value1 and ect_value2 and weat_effect_size and kmeans is None:
        response = jsonify(message="400 Error: Calculation failed please try again.")
        return response
    else:
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
