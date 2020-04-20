import json_controller
import upload_controller
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


def retrieve_uploaded_vector_single(space, target):
    found = {}
    not_found = []
    if space == upload_controller.uploaded_filename:
        if target in upload_controller.uploaded_space:
            found[target] = upload_controller.uploaded_space[target]
        else:
            not_found.append(target)
    return found, not_found


def retrieve_uploaded_vector_multiple(space, word_list):
    not_found = []
    found = {}
    if space == upload_controller.uploaded_filename:
        for word in word_list:
            if word in upload_controller.uploaded_space:
                found[word] = upload_controller.uploaded_space[word]
            else:
                not_found.append(word)
    return found, not_found


def retrieve_vector(number, content, bar):
    vectors = {}
    not_found = []

    if 'space' not in bar:
        return 'BAD REQUEST - NO SPACE SELECTED', 400
    space = bar['space']
    uploaded = 'false'
    lower = 'false'
    if 'uploaded' in bar:
        uploaded = bar['uploaded']
    if 'lower' in bar:
        lower = bar['lower']

    if number == 'single':
        if 'word' not in bar:
            return 'BAD REQUEST - NO WORD INPUT', 400
        target = bar['word']
        if lower == 'true':
            target = target.lower()
        if uploaded == 'true':
            vectors, not_found = retrieve_uploaded_vector_single(space, target)
        if space == 'fasttext':
            vectors, not_found = retrieve_vector_single(ft_vocab, ft_vecs, target)
        if space == 'glove':
            vectors, not_found = retrieve_vector_single(gv_vocab, gv_vecs, target)
        if space == 'cbow':
            vectors, not_found = retrieve_vector_single(cb_vocab, cb_vecs, target)

    if number == 'multiple':
        if content is None:
            return 'BAD REQUEST - NO WORD LIST RECEIVED', 400
        target = content['Words'].split(' ')
        if lower == 'true':
            target = [t.lower() for t in target]
        if uploaded == 'true':
            vectors, not_found = retrieve_uploaded_vector_multiple(space, target)
        if space == 'fasttext':
            vectors, not_found = retrieve_vector_multiple(ft_vocab, ft_vecs, target)
        if space == 'glove':
            vectors, not_found = retrieve_vector_multiple(gv_vocab, gv_vecs, target)
        if space == 'cbow':
            vectors, not_found = retrieve_vector_multiple(cb_vocab, cb_vecs, target)

    if vectors and not_found is not None:
        response = json_controller.json_vector_retrieval(vectors, not_found)
        return response, 200
    else:
        return "NOT FOUND", 404


