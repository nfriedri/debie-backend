import json
import logging

from flask import jsonify

import JSONFormatter
import calculation
from debiasing import bam2, gbdd


def return_full_debiasing(methods, arguments, content):
    logging.info("APP-DE: Forwarding to related definitions")
    database = arguments['space']
    augment_flag = arguments['augments']
    target1, target2, aug1, aug2 = JSONFormatter.retrieve_vectors_debiasing(content, database, augment_flag)
    target1, target2 = calculation.check_sizes(target1, target2)
    aug1, aug2 = calculation.check_sizes(aug1, aug2)
    logging.info("APP: Final retrieved set sizes: T1=" + str(len(target1)) + " T2=" + str(len(target2)) + " A1=" + str(
        len(aug1)) + " A2=" + str(len(aug2)))
    if len(target1) == 0 or len(target2) == 0 or len(aug1) == 0 or len(aug2) == 0:
        logging.info("APP: Stopped, no values found in database")
        return jsonify(message="ERROR: No values found in database."), 404
    logging.info("APP: Debiasing process started")
    result1, result2 = {}, {}
    try:
        if methods is None:
            result1, result2 = gbdd.generalized_bias_direction_debiasing(target1, target2, aug1, aug2)
        if methods == 'gbdd':
            result1, result2 = gbdd.generalized_bias_direction_debiasing(target1, target2, aug1, aug2)
        if methods == 'bam':
            result1, result2 = bam2.bias_alignment_model(target1, target2, aug1, aug2)
        if methods == 'gbddxbam':
            result1, result2 = gbdd.generalized_bias_direction_debiasing(target1, target2, aug1, aug2)
            result1, result2 = bam2.bias_alignment_model(result1, result2, aug1, aug2)
        if methods == 'bamxgbdd':
            result1, result2 = bam2.bias_alignment_model(target1, target2, aug1, aug2)
            result1, result2 = gbdd.generalized_bias_direction_debiasing(result1, result2, aug1, aug2)
        biased_terms = calculation.concatenate_dicts(target1, target2)
        debiased_terms = calculation.concatenate_dicts(result1, result2)
        response = json.dumps(
            {"EmbeddingSpace": database, "Method": methods,
             "BiasedVecs:": JSONFormatter.dict_to_json(biased_terms),
             "DebiasedVecs": JSONFormatter.dict_to_json(debiased_terms)})
    except:
        return jsonify(message="DEBIASING ERROR"), 500
    logging.info("APP: Debiasing process finished")
    return response, 200


def return_pca_debiasing(methods, arguments, content):
    logging.info("APP-DE: Forwarding to related definitions")
    database = arguments['space']
    augment_flag = arguments['augments']
    target1, target2, aug1, aug2 = JSONFormatter.retrieve_vectors_debiasing(content, database, augment_flag)
    target1, target2 = calculation.check_sizes(target1, target2)
    aug1, aug2 = calculation.check_sizes(aug1, aug2)
    logging.info("APP: Final retrieved set sizes: T1=" + str(len(target1)) + " T2=" + str(len(target2)) + " A1=" + str(
        len(aug1)) + " A2=" + str(len(aug2)))
    if len(target1) == 0 or len(target2) == 0 or len(aug1) == 0 or len(aug2) == 0:
        logging.info("APP: Stopped, no values found in database")
        return jsonify(message="ERROR: No values found in database."), 404
    logging.info("APP: Debiasing process started")
    result1, result2 = {}, {}
    try:
        if methods is None:
            result1, result2 = gbdd.generalized_bias_direction_debiasing(target1, target2, aug1, aug2)
        if methods == 'gbdd':
            result1, result2 = gbdd.generalized_bias_direction_debiasing(target1, target2, aug1, aug2)
        if methods == 'bam':
            result1, result2 = bam2.bias_alignment_model(target1, target2, aug1, aug2)
        if methods == 'gbddxbam':
            result1, result2 = gbdd.generalized_bias_direction_debiasing(target1, target2, aug1, aug2)
            result1, result2 = bam2.bias_alignment_model(result1, result2, aug1, aug2)
        if methods == 'bamxgbdd':
            result1, result2 = bam2.bias_alignment_model(target1, target2, aug1, aug2)
            result1, result2 = gbdd.generalized_bias_direction_debiasing(result1, result2, aug1, aug2)
        target1_copy, target2_copy = calculation.create_duplicates(target1, target2)
        result1_copy, result2_copy = calculation.create_duplicates(result1, result2)
        biased_terms = calculation.concatenate_dicts(target1_copy, target2_copy)
        debiased_terms = calculation.concatenate_dicts(result1_copy, result2_copy)
        biased_pca = calculation.principal_componant_analysis(target1, target2)
        debiased_pca = calculation.principal_componant_analysis(result1, result2)
        response = json.dumps(
            {"EmbeddingSpace": database, "Method": methods,
             "BiasedVectorsPCA": JSONFormatter.dict_to_json(biased_pca),
             "DebiasedVectorsPCA": JSONFormatter.dict_to_json(debiased_pca),
             "BiasedVecs:": JSONFormatter.dict_to_json(biased_terms),
             "DebiasedVecs": JSONFormatter.dict_to_json(debiased_terms)})
    except:
        return jsonify(message="DEBIASING ERROR"), 500
    logging.info("APP: Debiasing process finished")
    return response, 200

