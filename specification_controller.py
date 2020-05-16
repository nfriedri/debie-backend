import logging
import random

import numpy

import upload_controller
from vector_retrieval import retrieve_vector_multiple
from vector_retrieval import retrieve_uploaded_vector_multiple

from data_controller import fasttext_vocab as ft_vocab
from data_controller import fasttext_vectors as ft_vecs
from data_controller import glove_vocab as gv_vocab
from data_controller import glove_vectors as gv_vecs
from data_controller import cbow_vocab as cb_vocab
from data_controller import cbow_vectors as cb_vecs


def format_sets(t1, t2, a1, a2):
    t1, t2, t_del = format_set_sizes(t1, t2)
    a1, a2, a_del = format_set_sizes(a1, a2)
    deleted_keys = t_del + a_del
    return t1, t2, a1, a2, deleted_keys


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
                logging.info("SpecController: Removed keys from dictionary 1: " + str(key))
        elif len(vector_set2) > len(vector_set1):
            difference = len(vector_set2) - len(vector_set1)
            for i in range(difference):
                key = random.choice(list(vector_set2.keys()))
                del vector_set2[key]
                deleted_keys.append(key)
                logging.info("SpecController: Removed keys from dictionary 2: " + str(key))
        return vector_set1, vector_set2, deleted_keys


def get_vectors_for_spec(space, lower, uploaded, t1, t2, a1, a2, aug1=None, aug2=None):
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

    if uploaded == 'true':
        t1_found, t1_not_found = retrieve_uploaded_vector_multiple(space, t1)
        t2_found, t2_not_found = retrieve_uploaded_vector_multiple(space, t2)
        a1_found, a1_not_found = retrieve_uploaded_vector_multiple(space, a1)
        a2_found, a2_not_found = retrieve_uploaded_vector_multiple(space, a2)
        not_found = t1_not_found + t2_not_found + a1_not_found + a2_not_found
        if aug1 is not None and aug2 is not None:
            aug1_found, aug1_not_found = retrieve_uploaded_vector_multiple(space, aug1)
            aug2_found, aug2_not_found = retrieve_uploaded_vector_multiple(space, aug2)
            not_found += aug1_not_found + aug2_not_found
    t1, t2, t_del = format_set_sizes(t1_found, t2_found)
    a1, a2, a_del = format_set_sizes(a1_found, a2_found)
    deleted_keys = t_del + a_del
    if aug1 is not None and aug2 is not None:
        aug1, aug2, aug_del = format_set_sizes(aug1_found, aug2_found)
        deleted_keys += aug_del
        logging.info("SpecController: Returning found vectors")
        logging.info("SpecController: NotFound: " + str(not_found) + " ; DeletedKeys: " + str(deleted_keys))
        return t1, t2, a1, a2, aug1, aug2, not_found, deleted_keys

    return t1, t2, a1, a2, not_found, deleted_keys


def string_dicts_to_numpy_array_dicts(t1, t2, a1, a2):
    target1, target2, attribute1, attribute2 = {}, {}, {}, {}
    for value in t1:
        val = numpy.array(t1[value])
        target1[value] = val.astype(numpy.float)
    for value in t2:
        val = numpy.array(t2[value])
        target2[value] = val.astype(numpy.float)
    for value in a1:
        val = numpy.array(a1[value])
        attribute1[value] = val.astype(numpy.float)
    for value in a2:
        val = numpy.array(a2[value])
        attribute2[value] = val.astype(numpy.float)
    target1, target2, attribute1, attribute2, deleted = format_sets(target1, target2, attribute1, attribute2)
    return target1, target2, attribute1, attribute2, deleted


def get_vectors_for_augments(space, lower, uploaded, aug1_list, aug2_list):
    found1, found2 = {}, {}
    not_found1, not_found2 = [], []
    if lower == 'true':
        aug1_list = [x.lower() for x in aug1_list]
        aug2_list = [x.lower() for x in aug2_list]
    if space == 'fasttext':
        found1, not_found1 = retrieve_vector_multiple(ft_vocab, ft_vecs, aug1_list)
        found2, not_found2 = retrieve_vector_multiple(ft_vocab, ft_vecs, aug2_list)
    if space == 'glove':
        found1, not_found1 = retrieve_vector_multiple(gv_vocab, gv_vecs, aug1_list)
        found2, not_found2 = retrieve_vector_multiple(gv_vocab, gv_vecs, aug2_list)
    if space == 'cbow':
        found1, not_found1 = retrieve_vector_multiple(cb_vocab, cb_vecs, aug1_list)
        found2, not_found2 = retrieve_vector_multiple(cb_vocab, cb_vecs, aug2_list)
    if uploaded == 'true':
        found1, not_found1 = retrieve_uploaded_vector_multiple(space, aug1_list)
        found2, not_found2 = retrieve_uploaded_vector_multiple(space, aug2_list)

    not_found = not_found1 + not_found2
    aug1, aug2, deleted = format_set_sizes(found1, found2)

    return aug1, aug2, not_found, deleted


def return_vocab_vecs(space, uploaded):
    vocab = {}
    vecs = []
    if space == 'fasttext':
        print("here should i be")
        return ft_vocab, ft_vecs
    if space == 'glove':
        return gv_vocab, gv_vecs
    if space == 'cbow':
        return cb_vocab, cb_vecs
    if uploaded == 'true':
        return upload_controller.get_vocab_vecs_from_upload()
    else:
        return vocab, vecs
