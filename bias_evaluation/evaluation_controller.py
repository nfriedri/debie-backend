import augmentation_retrieval
import json_controller
import specification_controller
import logging
import calculation
from bias_evaluation import ect, bat, weat, kmeans, svm_classifier, semantic_quality


def evaluation(methods, content, bar):
    logging.info("Eval-Engine: Evaluation started")
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
    if json == 'true':
        t1, t2, a1, a2, aug1, aug2 = json_controller.json_with_vector_data(content)
        t1, t2, a1, a2, deleted = specification_controller.string_dicts_to_numpy_array_dicts(t1, t2, a1, a2)
        not_found = []
    else:
        t1, t2, a1, a2 = json_controller.json_to_bias_spec(content)
        t1, t2, a1, a2, not_found, deleted = specification_controller.get_vectors_for_spec(space, lower, uploaded, t1, t2, a1, a2)
    scores = {}
    if methods is None:
        scores = evaluate_all(space, lower, uploaded, t1, t2, a1, a2)
    if methods == 'all':
        scores = evaluate_all(space, lower, uploaded, t1, t2, a1, a2)
    if methods == 'ect':
        scores = evaluate_ect(t1, t2, a1, a2)
    if methods == 'bat':
        scores = evaluate_bat(t1, t2, a1, a2)
    if methods == 'weat':
        scores = evaluate_weat(t1, t2, a1, a2)
    if methods == 'kmeans':
        scores = evaluate_kmeans(t1, t2)
    if methods == 'svm':
        scores = evaluate_svm(space, lower, uploaded, t1, t2)
    if methods == 'simlex':
        scores = evaluate_simlex(space, uploaded)
    if methods == 'wordsim':
        scores = evaluate_wordsim(space, uploaded)
    return json_controller.bias_evaluation_json(scores, space, lower, t1, t2, a1, a2, not_found, deleted)


def evaluate_all(space, lower, uploaded, t1, t2, a1, a2):
    logging.info("APP-BE: Starting all evaluations")
    ect_score, ect_p_value = ect.embedding_coherence_test(t1, t2, a1, a2)
    bat_score = bat.bias_analogy_test(t1, t2, a1, a2)
    weat_effect_size, weat_p_value = weat.word_embedding_association_test(t1, t2, a1, a2)
    k_means = kmeans.k_means_clustering(t1, t2)
    svm = evaluate_svm(space, lower, uploaded, t1, t2)
    logging.info("APP-BE: Evaluations finished successfully")
    scores = {'ECT_Score': ect_score, 'ECT_P_Value': ect_p_value, 'BAT_Score': bat_score,
              'WEAT_Effect_Size': weat_effect_size, 'WEAT_P_Value': weat_p_value, 'K_Means': k_means}
    scores.update(svm)
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


def evaluate_kmeans(t1, t2):
    k_means = kmeans.k_means_clustering(t1, t2)
    scores = {'K_Means': k_means}
    return scores


def evaluate_svm(space, lower, uploaded, t1, t2):
    t1_list = list(t1.keys())
    t2_list = list(t2.keys())
    augments1_list, computed1 = augmentation_retrieval.retrieve_multiple_augmentations(t1_list)
    augments2_list, computed2 = augmentation_retrieval.retrieve_multiple_augmentations(t2_list)
    aug1, aug2, not_found, deleted = specification_controller.get_vectors_for_augments(space, lower, uploaded, augments1_list, augments2_list)
    vocab, vecs = calculation.create_vocab_and_vecs(t1, t2, aug1, aug2)
    svm = svm_classifier.eval_svm(augments1_list, augments2_list, t1_list, t2_list, vocab, vecs)
    scores = {'SVM': svm}
    return scores


def evaluate_simlex(space, uploaded):
    vocab, vecs = specification_controller.return_vocab_vecs(space, uploaded)
    pearson, spearman = semantic_quality.eval_simlex(vocab, vecs, 'SimLex')
    scores = {'SimLexPearson': pearson, 'SimLexSpearman': spearman}
    return scores


def evaluate_wordsim(space, uploaded):
    vocab, vecs = specification_controller.return_vocab_vecs(space, uploaded)
    pearson, spearman = semantic_quality.eval_simlex(vocab, vecs, 'WordSim')
    scores = {'WordSimPearson': pearson, 'WordSimSpearman': spearman}
    return scores
