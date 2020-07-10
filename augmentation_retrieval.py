import json_controller
import calculation
import numpy as np

import upload_controller
from data_controller import augmentations
from data_controller import fasttext_vocab as ft_vocab
from data_controller import fasttext_vectors as ft_vecs
from data_controller import glove_vocab as gv_vocab
from data_controller import glove_vectors as gv_vecs
from data_controller import cbow_vocab as cb_vocab
from data_controller import cbow_vectors as cb_vecs


def retrieve_augmentations(number, content, bar):
    augments = {}
    computed_augments = []

    uploaded = 'false'
    lower = 'false'
    iterations = 3
    space = 'fasttext'
    if 'uploaded' in bar:
        uploaded = bar['uploaded']
    if 'lower' in bar:
        lower = bar['lower']
    if 'iterations' in bar:
        iterations = bar['iterations']
    if 'space' in bar:
        space = bar['space']

    if number == 'single':
        if 'word' not in bar:
            return 'BAD REQUEST - NO WORD INPUT', 400
        target = bar['word']
        if lower == 'true':
            target.lower()
        if space == 'glove':
            augments[target], computed = retrieve_single_augmentations(target, gv_vocab, gv_vecs)
        if space == 'cbow':
            augments[target], computed = retrieve_single_augmentations(target, cb_vocab, cb_vecs)
        if uploaded == 'true' and upload_controller.uploaded_binary == 'true':
            augments[target], computed = retrieve_single_augmentations(target, upload_controller.uploaded_vocab,
                                                                       upload_controller.uploaded_vecs)
        else:
            augments[target], computed = retrieve_single_augmentations(target)
        if computed:
            computed_augments.append(target)
    if number == 'multiple':
        if content is None:
            return 'BAD REQUEST - NO WORD INPUT', 400
        target = content['Words'].split(' ')
        if lower == 'true':
            target = [x.lower() for x in target]
        vocab = ft_vocab
        vecs = ft_vecs
        if space == 'glove':
            vocab = gv_vocab
            vecs = gv_vecs
        if space == 'cbow':
            vocab = cb_vocab
            vecs = cb_vecs
        if uploaded == 'true' and upload_controller.uploaded_binary == 'true':
            vocab = upload_controller.uploaded_vocab
            vecs = upload_controller.uploaded_vecs
        for word in target:
            augments[word], computed = retrieve_single_augmentations(word, vocab, vecs)
            if computed:
                computed_augments.append(word)
    response = json_controller.json_augmentation_retrieval(augments, computed_augments)
    return response, 200


def retrieve_single_augmentations(target, vocab=ft_vocab, vecs=ft_vecs):
    if target in augmentations:
        return augmentations[target], False
    # TODO elif: search whether augmentation has already been computed once, if yes retrieve it
    # TODO inform user whether augments are postspecialized or not
    else:
        augments, computed = compute_augmentations(target, vocab, vecs)
        return augments, computed


def retrieve_multiple_augmentations(target):
    augments = []
    computed_augments = []
    for word in target:
        aug, computed = retrieve_single_augmentations(word)
        for w in aug:
            augments.append(w)
        if computed:
            computed_augments.append(word)
    return augments, computed_augments


def compute_augmentations(target, vocab, vecs, iterations=4):
    # print('COMPUTED')
    augments = []
    cosinesim = {}
    if target not in vocab:
        return 'Not in vocab', False
    target_vec = np.array(vecs[vocab[target]])
    for word in vocab:
        vec = np.array(vecs[vocab[word]])
        if word != target:
            cosinesim[word] = calculation.cosine_similarity(target_vec, vec)
    for i in range(iterations):
        maximum = max(cosinesim, key=lambda k: cosinesim[k])
        cosinesim.pop(maximum)
        augments.append(maximum)
    return augments, True
