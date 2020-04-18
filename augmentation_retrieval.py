import json_controller
from data_controller import augmentations
from data_controller import fasttext_vocab
from data_controller import fasttext_vectors
import calculation
import numpy as np


def retrieve_augmentations(number, content, bar):
    augments = {}
    computed_augments = []

    uploaded = 'false'
    lower = 'false'
    iterations = 4
    if 'uploaded' in bar:
        uploaded = bar['uploaded']
    if 'lower' in bar:
        lower = bar['lower']
    if 'iterations' in bar:
        iterations = bar['iterations']

    if number == 'single':
        if 'word' not in bar:
            return 'BAD REQUEST - NO WORD INPUT', 400
        target = bar['word']
        if lower == 'true':
            target.lower()
        augments[target], computed = retrieve_single_augmentations(target)
        if computed:
            computed_augments.append(target)
    if number == 'multiple':
        if content is None:
            return 'BAD REQUEST - NO WORD INPUT', 400
        target = content['Words'].split(' ')
        if lower == 'true':
            target = [x.lower() for x in target]
        for word in target:
            augments[word], computed = retrieve_single_augmentations(word)
            if computed:
                computed_augments.append(word)
    response = json_controller.json_augmentation_retrieval(augments, computed_augments)
    return response, 200


def retrieve_single_augmentations(target):
    if target in augmentations:
        return augmentations[target], False
    # TODO elif: search whether augmentation has already been computed once, if yes retrieve it
    # TODO inform user whether augments are postspecialized or not
    else:
        augments, computed = compute_augmentations(target)
        return augments, computed


def compute_augmentations(target, vocab=fasttext_vocab, vecs=fasttext_vectors, iterations=4):
    print('COMPUTED')
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
