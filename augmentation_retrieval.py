from data_controller import augmentations
from data_controller import fasttext_vocab
from data_controller import fasttext_vectors
import calculation
import numpy as np


def retrieve_augmentations(target):
    if target in augmentations:
        return augmentations[target]
    # TODO elif: search whether augmentation has already been computed once, if yes retrieve it
    # TODO inform user whether augments are postspecialized or not
    else:
        augments = compute_augmentations(target)
        # save_new_augmentations(augments)
        return augments


def compute_augmentations(target, vocab=fasttext_vocab, vecs=fasttext_vectors, iterations=4):
    augments = []
    cosinesim = {}
    if target not in vocab:
        return 'Not on vocab'
    target_vec = np.array(vecs[vocab[target]])
    for word in vocab:
        vec = np.array(vecs[vocab[word]])
        if word != target:
            cosinesim[word] = calculation.cosine_similarity(target_vec, vec)
    for i in range(iterations):
        maximum = max(cosinesim, key=lambda k: cosinesim[k])
        cosinesim.pop(maximum)
        augments.append(maximum)
    return augments
