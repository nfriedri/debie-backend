import data_controller
from data_controller import fasttext_vocab as ft_vocab
from data_controller import fasttext_vectors as ft_vecs
from data_controller import glove_vocab as gv_vocab
from data_controller import glove_vectors as gv_vecs
from data_controller import cbow_vocab as cb_vocab
from data_controller import cbow_vectors as cb_vecs


def retrieve_vector_single(vocab, vectors, target_word):
    found = {}
    not_found = []
    if target_word in vocab:
        found[target_word] = vectors[vocab[target_word]]
    else:
        not_found.append(target_word)

    return found, not_found


def retrieve_vector_multiple(vocab, vectors, word_list):
    not_found = []
    found = {}
    for word in word_list:
        if word in vocab:
            found[word] = vectors[vocab[word]]
        else:
            not_found.append(word)
    return found, not_found


def retrieve_vector(number, uploaded, embeddings, target, lower):
    if number == 'single':
        if lower == 'true':
            target = target.lower()
        if embeddings == 'fasttext':
            return retrieve_vector_single(ft_vocab, ft_vecs, target)
        if embeddings == 'glove':
            return retrieve_vector_single(gv_vocab, gv_vecs, target)
        if embeddings == 'cbow':
            return retrieve_vector_single(cb_vocab, cb_vecs, target)

    if number == 'multiple':
        if lower == 'true':
            target = [t.lower() for t in target]
        if embeddings == 'fasttext':
            return retrieve_vector_multiple(ft_vocab, ft_vecs, target)
        if embeddings == 'glove':
            return retrieve_vector_multiple(gv_vocab, gv_vecs, target)
        if embeddings == 'cbow':
            return retrieve_vector_multiple(cb_vocab, cb_vecs, target)
        # if uploaded is 'true':
        # return 'Solution needed', 'Rework required'
        # else:
        # return 'error', 'error'
