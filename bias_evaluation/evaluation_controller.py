import json_controller
import specification_controller

from bias_evaluation import ect, bat, weat, kmeans


def evaluation(methods, content, bar):
    print("Bias Eval")
    if content is None:
        return 'BAD REQUEST - NO BIAS SPEC JSON FOUND', 400
    if 'space' not in bar and 'uploaded' not in bar:
        return 'BAD REQUEST - NO EMBEDDING SPACE SELECTED', 400
    space = 'fasttext'
    uploaded = 'false'
    lower = 'false'
    json = 'false'
    if 'space' in bar:
        space = bar['space']
    if 'uploaded' in bar:
        uploaded = bar['uploaded']
    if 'lower' in bar:
        lower = bar['lower']
    if 'json' in bar:
        json = bar['json']

    print('searching error 1')
    if json == 'true':
        t1, t2, a1, a2, aug1, aug2 = json_controller.json_with_vector_data(content)
        print('problem')
        t1, t2, a1, a2, deleted = specification_controller.string_dicts_to_numpy_array_dicts(t1, t2, a1, a2)
        not_found = []
    else:
        t1, t2, a1, a2 = json_controller.json_to_bias_spec(content)
        t1, t2, a1, a2, not_found, deleted = specification_controller.get_vectors_for_spec(space, lower, uploaded, t1, t2, a1, a2)
    scores = {}
    print("Retrieved SPecs")
    if methods is None:
        scores = evaluate_all(t1, t2, a1, a2)
    if methods == 'all':
        scores = evaluate_all(t1, t2, a1, a2)
    if methods == 'ect':
        scores = evaluate_ect(t1, t2, a1, a2)
    if methods == 'bat':
        scores = evaluate_bat(t1, t2, a1, a2)
    if methods == 'weat':
        scores = evaluate_weat(t1, t2, a1, a2)
    if methods == 'kmeans':
        scores = evaluate_kmeans(t1, t2, a1, a2)
    return json_controller.bias_evaluation_json(scores, space, lower, t1, t2, a1, a2, not_found, deleted)


def evaluate_all(t1, t2, a1, a2):
    # logging.info("APP-BE: Starting all evaluations")
    ect_score, ect_p_value = ect.embedding_coherence_test(t1, t2, a1, a2)
    bat_score = bat.bias_analogy_test(t1, t2, a1, a2)
    weat_effect_size, weat_p_value = weat.word_embedding_association_test(t1, t2, a1, a2)
    k_means = kmeans.k_means_clustering(t1, t2)
    # logging.info("APP-BE: Evaluations finished successfully")
    scores = {'ECT_Score': ect_score, 'ECT_P_Value': ect_p_value, 'BAT_Score': bat_score,
              'WEAT_Effect_Size': weat_effect_size, 'WEAT_P_Value': weat_p_value, 'K_Means': k_means}
    return scores


def evaluate_ect(t1, t2, a1, a2):
    ect_score, ect_p_value = ect.embedding_coherence_test(t1, t2, a1, a2)
    scores = {'ECT_Score': ect_score, 'ECT_P_Value': ect_p_value}
    return scores


def evaluate_bat(t1, t2, a1, a2):
    bat_score = bat.bias_analogy_test(t1, t2, a1, a2)
    scores = {'BAT_Score': bat_score}
    return scores


def evaluate_weat(t1, t2, a1, a2):
    weat_effect_size, weat_p_value = weat.word_embedding_association_test(t1, t2, a1, a2)
    scores = {'WEAT_Effect_Size': weat_effect_size, 'WEAT_P_Value': weat_p_value}
    return scores


def evaluate_kmeans(t1, t2, a1, a2):
    k_means = kmeans.k_means_clustering(t1, t2)
    scores = {'K_Means': k_means}
    return scores
