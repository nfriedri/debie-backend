import specification_controller
import augmentation_retrieval
import json_controller
import numpy as np
import calculation

from debiasing import bam, gbdd


def debiasing(methods, content, bar):
    # bar params: lower, uploaded, pca, space
    if content is None:
        return 'BAD REQUEST - NO BIAS SPEC JSON FOUND', 400
    if 'space' not in bar:
        return 'BAD REQUEST - NO SPACE SELECTED', 400
    space = bar['space']
    uploaded = 'false'
    lower = 'false'
    pca = 'true'
    if 'uploaded' in bar:
        uploaded = bar['uploaded']
    if 'lower' in bar:
        lower = bar['lower']
    if 'pca' in bar:
        pca = bar['pca']

    t1_list, t2_list, a1_list, a2_list, aug1_list, aug2_list, not_found, deleted = json_controller.json_to_debias_spec(content)
    if len(aug1_list) == 0:
        aug1 = augmentation_retrieval.retrieve_multiple_augmentations(a1_list)
    if len(aug2_list) == 0:
        aug2 = augmentation_retrieval.retrieve_multiple_augmentations(a2_list)

    equality_sets = []
    for i in range(len(aug1_list)):
        for j in range(len(aug2_list)):
            equality_sets.append([aug1_list[i], aug2_list[j]])
    print(np.shape(equality_sets))

    t1, t2, a1, a2, aug1, aug2 = specification_controller.\
        get_vectors_for_spec(space, lower, t1_list, t2_list, a1_list, a2_list, aug1_list, aug2_list)
    vocab, vecs = create_vocab_and_vecs(t1, t2, a1, a2, aug1, aug2)
    t1_deb, t2_deb, a1_deb, a2_deb = [], [], [], []
    if methods == 'bam':
        t1_deb, t2_deb, a1_deb, a2_deb = debiasing_bam(equality_sets, vocab, vecs, t1_list, t2_list, a1_list, a2_list)
    if methods == 'gbdd':
        t1_deb, t2_deb, a1_deb, a2_deb = debiasing_gbdd(equality_sets, vocab, vecs, t1_list, t2_list, a1_list, a2_list)
    if methods == 'bamXgbdd':
        t1_deb, t2_deb, a1_deb, a2_deb = debiasing_bam_gbdd(equality_sets, vocab, vecs, t1_list, t2_list, a1_list, a2_list)
    if methods == 'gbddXbam':
        t1_deb, t2_deb, a1_deb, a2_deb = debiasing_gbdd_bam(equality_sets, vocab, vecs, t1_list, t2_list, a1_list, a2_list)

    if pca == 'true':
        t1_pca_bias, t2_pca_bias, a1_pca_bias, a2_pca_bias = calculation.principal_componant_analysis(t1, t2, a1, a2)
        t1_pca_deb, t2_pca_deb, a1_pca_deb, a2_pca_deb = calculation.principal_componant_analysis(t1_deb, t2_deb, a1_deb, a2_deb)
        response = json_controller.debiasisng_json(space, lower, methods, pca, aug1_list, aug2_list, t1, t2, a1, a2,
                                                   t1_deb, t2_deb, a1_deb, a2_deb, not_found, deleted,
                                                   t1_pca_bias, t2_pca_bias, a1_pca_bias, a2_pca_bias,
                                                   t1_pca_deb, t2_pca_deb, a1_pca_deb, a2_pca_deb)
    else:
        response = json_controller.debiasisng_json(space, lower, methods, pca, aug1_list, aug2_list, t1, t2, a1, a2,
                                                   t1_deb, t2_deb, a1_deb, a2_deb, not_found, deleted)
    return response


def debiasing_bam(equality_sets, vocab, vecs, t1_list, t2_list, a1_list, a2_list):
    new_vocab, new_vecs = bam.debias_proc(equality_sets, vocab, vecs)
    t1, t2, a1, a2 = vocab_to_dicts(new_vocab, new_vecs, t1_list, t2_list, a1_list, a2_list)
    return t1, t2, a1, a2


def debiasing_gbdd(equality_sets, vocab, vecs, t1_list, t2_list, a1_list, a2_list):
    v_b = gbdd.get_bias_direction(equality_sets, vecs, vocab)
    new_vocab, new_vecs = gbdd.debias_direction_linear(v_b, vecs)
    t1, t2, a1, a2 = vocab_to_dicts(new_vocab, new_vecs, t1_list, t2_list, a1_list, a2_list)
    return t1, t2, a1, a2


def debiasing_bam_gbdd(equality_sets, vocab, vecs, t1_list, t2_list, a1_list, a2_list):
    new_vocab, new_vecs = bam.debias_proc(equality_sets, vocab, vecs)
    v_b = gbdd.get_bias_direction(equality_sets, new_vecs, new_vocab)
    new_vocab, new_vecs = gbdd.debias_direction_linear(v_b, new_vecs)
    t1, t2, a1, a2 = vocab_to_dicts(new_vocab, new_vecs, t1_list, t2_list, a1_list, a2_list)
    return t1, t2, a1, a2


def debiasing_gbdd_bam(equality_sets, vocab, vecs, t1_list, t2_list, a1_list, a2_list):
    v_b = gbdd.get_bias_direction(equality_sets, vecs, vocab)
    new_vocab, new_vecs = gbdd.debias_direction_linear(v_b, vecs)
    new_vocab, new_vecs = bam.debias_proc(equality_sets, new_vocab, new_vecs)
    t1, t2, a1, a2 = vocab_to_dicts(new_vocab, new_vecs, t1_list, t2_list, a1_list, a2_list)
    return t1, t2, a1, a2


def create_vocab_and_vecs(t1, t2, a1, a2, aug1, aug2):
    vocab = {}
    vecs = []
    counter = 0
    dicts = t1.update(t2).update(a1).update(a2).update(aug1).update(aug2)
    for word in dicts:
        vocab[word] = counter
        vecs.append(dicts[word])
        counter += 1
    return vocab, vecs


def vocab_to_dicts(vocab, vecs, t1_list, t2_list, a1_list, a2_list):
    t1, t2, a1, a2 = {}, {}, {}, {}
    for word in t1_list:
        t1[word] = vecs[vocab[word]]
    for word in t2_list:
        t2[word] = vecs[vocab[word]]
    for word in a1_list:
        a1[word] = vecs[vocab[word]]
    for word in a2_list:
        a2[word] = vecs[vocab[word]]
    return t1, t2, a1, a2


