from flask import jsonify

import JSONFormatter
import calculation
from bias_evaluation import weat, ect, k_means, bat
import logging


def return_bias_evaluation(methods, arguments, content):
    logging.info("APP-BE: Forwarding to related definitions")
    database = 'fasttext'
    if 'space' in arguments.keys:
        database = arguments['space']
    vector_flag = 'false'
    if 'vectors' in arguments.keys():
        vector_flag = arguments['vectors']

    if vector_flag == 'false':
        target1, target2, arg1, arg2 = JSONFormatter.retrieve_vectors_evaluation(content, database)
    else:
        target1, target2, arg1, arg2 = JSONFormatter.retrieve_vectors_from_json(content)

    target1, target2 = calculation.check_sizes(target1, target2)
    arg1, arg2 = calculation.check_sizes(arg1, arg2)
    if len(target1) == 0 or len(target2) == 0:
        logging.info("APP: Stopped, no values found in database")
        return jsonify(message="ERROR: No values found in database."), 404
    if len(arg1) == 0 and len(arg2) == 0 and methods != 'kmeans':
        return jsonify(message="No attribute sets provided, only k means++ is executable"), 400
    logging.info("APP: Final retrieved set sizes: T1=" + str(len(target1)) + " T2=" + str(len(target2)) + " A1=" + str(
        len(arg1)) + " A2=" + str(len(arg2)))
    logging.info("APP: Evaluation process started")

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
        bat_result = bat.bias_analogy_test(test_vectors1, test_vectors2, arg_vectors1, arg_vectors2)
        # bat_result = 'Currently not available'
        weat_effect_size, weat_p_value = weat.word_embedding_association_test(test_vectors1, test_vectors2,
                                                                              arg_vectors1,
                                                                              arg_vectors2)
        kmeans = k_means.k_means_clustering(test_vectors1, test_vectors2)
        logging.info("APP-BE: Evaluations finished successfully")
        response = jsonify(ect_value1=ect_value1, p_value1=p_value1, p_value2=p_value2, ect_value2=ect_value2,
                           bat_result=bat_result, weat_effect_size=weat_effect_size, weat_pvalue=weat_p_value,
                           k_means=kmeans)
        logging.info("APP-BE: Results: " + str(response))
        return response
    except RuntimeWarning as rw:
        print(rw)

    return jsonify(message="Internal Calculation Error")


def return_eval_ect(test_vectors1, test_vectors2, arg_vectors1, arg_vectors2):
    logging.info("APP-BE: Starting ECT evaluation")
    ect_value1, p_value1 = ect.embedding_coherence_test(test_vectors1, test_vectors2, arg_vectors1)
    ect_value2, p_value2 = ect.embedding_coherence_test(test_vectors1, test_vectors2, arg_vectors2)
    logging.info("APP-BE: ECT finished successfully")
    response = jsonify(ect_value1=ect_value1, p_value1=p_value1, p_value2=p_value2,
                       ect_value2=ect_value2)
    logging.info("APP-BE: Results: " + str(response))
    return response


def return_eval_bat(test_vectors1, test_vectors2, arg_vectors1, arg_vectors2):
    logging.info("APP-BE: Starting BAT evaluation")
    bat_result = bat.bias_analogy_test(test_vectors1, test_vectors2, arg_vectors1, arg_vectors2)
    # bat_result = 'Currently not available'
    logging.info("APP-BE: BAT finished successfully")
    response = jsonify(bat_result=bat_result)
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
