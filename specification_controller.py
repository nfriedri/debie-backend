import random

from vector_retrieval import retrieve_vector_multiple

from data_controller import fasttext_vocab as ft_vocab
from data_controller import fasttext_vectors as ft_vecs
from data_controller import glove_vocab as gv_vocab
from data_controller import glove_vectors as gv_vecs
from data_controller import cbow_vocab as cb_vocab
from data_controller import cbow_vectors as cb_vecs


def format_set_sizes(vector_set1, vector_set2):
    deleted_keys = []
    if (len(vector_set1) > 0) & (len(vector_set2) > 0):
        if len(vector_set1) == len(vector_set2):
            return vector_set1, vector_set2, deleted_keys
        elif len(vector_set1) > len(vector_set2):
            difference = len(vector_set1) - len(vector_set2)
            for i in range(difference):
                key = random.choice(list(vector_set1.keys()))
                del vector_set1[key]
                deleted_keys.append(key)
                # logging.info("CM: Removed keys from dictionary 2: " + str(key))
        elif len(vector_set2) > len(vector_set1):
            difference = len(vector_set2) - len(vector_set1)
            for i in range(difference):
                key = random.choice(list(vector_set2.keys()))
                del vector_set2[key]
                deleted_keys.append(key)
                # logging.info("CM: Removed keys from dictionary 2: " + str(key))
        return vector_set1, vector_set2, deleted_keys


def get_vectors_for_spec(space, lower, t1, t2, a1, a2, aug1=None, aug2=None):
    t1_found, t2_found, a1_found, a2_found, aug1_found, aug2_found = {}, {}, {}, {}, {}, {}
    not_found = []
    if lower == 'true':
        t1 = [x.lower() for x in t1]
        t2 = [x.lower() for x in t2]
        a1 = [x.lower() for x in a1]
        a2 = [x.lower() for x in a2]
        if aug1 is not None and aug2 is not None:
            aug1 = [x.lower() for x in aug1]
            aug2 = [x.lower() for x in aug2]
    if space == 'fasttext':
        t1_found, t1_not_found = retrieve_vector_multiple(ft_vocab, ft_vecs, t1)
        t2_found, t2_not_found = retrieve_vector_multiple(ft_vocab, ft_vecs, t2)
        a1_found, a1_not_found = retrieve_vector_multiple(ft_vocab, ft_vecs, a1)
        a2_found, a2_not_found = retrieve_vector_multiple(ft_vocab, ft_vecs, a2)
        not_found = t1_not_found + t2_not_found + a1_not_found + a2_not_found
        if aug1 is not None and aug2 is not None:
            aug1_found, aug1_not_found = retrieve_vector_multiple(ft_vocab, ft_vecs, aug1)
            aug2_found, aug2_not_found = retrieve_vector_multiple(ft_vocab, ft_vecs, aug2)
            not_found += aug1_not_found + aug2_not_found
    if space == 'glove':
        t1_found, t1_not_found = retrieve_vector_multiple(gv_vocab, gv_vecs, t1)
        t2_found, t2_not_found = retrieve_vector_multiple(gv_vocab, gv_vecs, t2)
        a1_found, a1_not_found = retrieve_vector_multiple(gv_vocab, gv_vecs, a1)
        a2_found, a2_not_found = retrieve_vector_multiple(gv_vocab, gv_vecs, a2)
        not_found = t1_not_found + t2_not_found + a1_not_found + a2_not_found
        if aug1 is not None and aug2 is not None:
            aug1_found, aug1_not_found = retrieve_vector_multiple(gv_vocab, gv_vecs, aug1)
            aug2_found, aug2_not_found = retrieve_vector_multiple(gv_vocab, gv_vecs, aug2)
            not_found += aug1_not_found + aug2_not_found
    if space == 'cbow':
        t1_found, t1_not_found = retrieve_vector_multiple(cb_vocab, cb_vecs, t1)
        t2_found, t2_not_found = retrieve_vector_multiple(cb_vocab, cb_vecs, t2)
        a1_found, a1_not_found = retrieve_vector_multiple(cb_vocab, cb_vecs, a1)
        a2_found, a2_not_found = retrieve_vector_multiple(cb_vocab, cb_vecs, a2)
        not_found = t1_not_found + t2_not_found + a1_not_found + a2_not_found
        if aug1 is not None and aug2 is not None:
            aug1_found, aug1_not_found = retrieve_vector_multiple(cb_vocab, cb_vecs, aug1)
            aug2_found, aug2_not_found = retrieve_vector_multiple(cb_vocab, cb_vecs, aug2)
            not_found += aug1_not_found + aug2_not_found

    t1, t2, t_del = format_set_sizes(t1_found, t2_found)
    a1, a2, a_del = format_set_sizes(a1_found, a2_found)
    deleted_keys = t_del + a_del
    if aug1 is not None and aug2 is not None:
        aug1, aug2, aug_del = format_set_sizes(aug1_found, aug2_found)
        deleted_keys += aug_del

    return t1, t2, a1, a2, not_found, deleted_keys
