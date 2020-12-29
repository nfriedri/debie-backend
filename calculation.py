import logging

import numpy
import random
import copy
from sklearn.decomposition import PCA

from data_controller import simlex_vocab, wordsim_vocab


# Creates deep-copies of one ore more dictionaries
def create_duplicates(set1, set2=None, set3=None, set4=None):
    copy1 = copy.deepcopy(set1)
    if set2 is not None:
        copy2 = copy.deepcopy(set2)
        if set3 is not None:
            copy3 = copy.deepcopy(set3)
            if set4 is not None:
                copy4 = copy.deepcopy(set4)
                return copy1, copy2, copy3, copy4
            return copy1, copy2, copy3
        return copy1, copy2
    return copy1


# Transforms a dictionary into a list
def transform_dict_to_list(dict1):
    vector_list = []
    for word in dict1:
        vector_list.append(list(dict1[word]))
    return numpy.array(vector_list)


# Extracts vectors from  two, three or four dictionaries into vector lists
def transform_multiple_dicts_to_lists(dict1, dict2, dict3=None, dict4=None):
    vectors1 = transform_dict_to_list(dict1)
    vectors2 = transform_dict_to_list(dict2)
    if dict3 is not None:
        vectors3 = transform_dict_to_list(dict3)
        if dict4 is not None:
            vectors4 = transform_dict_to_list(dict4)
            return vectors1, vectors2, vectors3, vectors4
        return vectors1, vectors2, vectors3
    return vectors1, vectors2


# Checks if sizes of vectors are equal, deletes random elements from the larger dict if not
def check_sizes(vector_set1, vector_set2):
    if (len(vector_set1) > 0) & (len(vector_set2) > 0):
        if len(vector_set1) == len(vector_set2):
            return vector_set1, vector_set2
        elif len(vector_set1) > len(vector_set2):
            difference = len(vector_set1) - len(vector_set2)
            for i in range(difference):
                key = random.choice(list(vector_set1.keys()))
                del vector_set1[key]
                logging.info("CM: Removed keys from dictionary 2: " + str(key))
        elif len(vector_set2) > len(vector_set1):
            difference = len(vector_set2) - len(vector_set1)
            for i in range(difference):
                key = random.choice(list(vector_set2.keys()))
                del vector_set2[key]
                logging.info("CM: Removed keys from dictionary 2: " + str(key))
    return vector_set1, vector_set2


# Checks weather two vector sets have the same length
def check_set_sizes(vector_set1, vector_set2):
    if len(vector_set1) > 0 & len(vector_set2) > 0:
        if len(vector_set1) != len(vector_set2):
            make_set_size_equal(vector_set1, vector_set2)
    return vector_set1, vector_set2


# Removes random elements from the longer list until their equal
def make_set_size_equal(vector_set1, vector_set2):
    logging.info('CM: Making set sizes equal:')
    while len(vector_set1) != len(vector_set2):
        if len(vector_set1) > len(vector_set2):
            key = random.choice(list(vector_set1.keys()))
            del vector_set1[key]
            logging.info("CM: REMOVED KEY from list 1: " + str(key))
        if len(vector_set2) > len(vector_set1):
            key = random.choice(list(vector_set2.keys()))
            del vector_set2[key]
            logging.info("CM: REMOVED KEY from list 2: " + str(key))
    return vector_set1, vector_set2


# Checks if two sets contain duplicates and removes them
def check_set_content(vector_set1, vector_set2):
    duplicates = [word for word in vector_set1 if word in vector_set2]
    if not duplicates:
        for word in duplicates:
            vector_set1.remove(word)
            vector_set2.remove(word)
            # print(duplicates[word])
    return vector_set1, vector_set2


# Makes and dict containing a vector to an numpy array for easier calculation
def create_numpy_vector(vector_set):
    array = []
    for word in vector_set:
        vector = list(vector_set[word])
        array.append(numpy.array(vector))
    numpy_array = numpy.array(array)
    return numpy_array


# Calculates the cosines similarity of two vectors
def cosine_similarity(vector1, vector2):
    dot = numpy.dot(vector1, vector2)
    norm_target = numpy.linalg.norm(vector1)
    norm_argument = numpy.linalg.norm(vector2)
    cos = dot / (norm_target * norm_argument)
    return cos


# Concatenates two dictionaries to one big
def concatenate_dicts(dict1, dict2, dict3=None, dict4=None):
    result = {}
    for word in dict1:
        result[word] = dict1[word]
    for word in dict2:
        result[word] = dict2[word]
    if dict3 is not None:
        for word in dict3:
            result[word] = dict3[word]
    if dict4 is not None:
        for word in dict4:
            result[word] = dict4[word]
    return result


# Calculates the euclidean distance between two vectors
def euclidean_distance(vector1, vector2):
    vector_a = numpy.array(vector1)
    vector_b = numpy.array(vector2)
    distance = numpy.linalg.norm(vector_a-vector_b)
    return distance


def create_vocab_and_vecs(t1, t2=None, a1=None, a2=None, aug1=None, aug2=None, lex_dict=None):
    vocab = {}
    vecs = []
    counter = 0
    dicts = {}
    dicts.update(t1)
    if t2 is not None:
        dicts.update(t2)
    if a1 is not None:
        dicts.update(a1)
    if a2 is not None:
        dicts.update(a2)
    if aug1 is not None:
        dicts.update(aug1)
    if aug2 is not None:
        dicts.update(aug2)
    if lex_dict is not None:
        dicts.update(lex_dict)
    for word in dicts:
        vocab[word] = counter
        vecs.append(dicts[word])
        counter += 1
    return vocab, vecs


def vocab_to_dict(vocab, vecs, lists):
    res = {}
    for word in lists:
        if word in vocab:
            res[word] = vecs[vocab[word]]
    return res


def vocabs_to_dicts(vocab, vecs, t1_list, t2_list, a1_list, a2_list, aug1_list=None, aug2_list=None, lex=None):
    t1 = vocab_to_dict(vocab, vecs, t1_list)
    t2 = vocab_to_dict(vocab, vecs, t2_list)
    a1 = vocab_to_dict(vocab, vecs, a1_list)
    a2 = vocab_to_dict(vocab, vecs, a2_list)
    if lex is not None:
        lex_dict = {}
        if lex == 'simlex':
            # print('Vocab to dicts --- SIMLEX')
            lex_dict = vocab_to_dict(vocab, vecs, simlex_vocab)
        if lex == 'wordsim':
            # print('Vocab to dicts --- WORDSIM')
            lex_dict = vocab_to_dict(vocab, vecs, wordsim_vocab)
        return t1, t2, a1, a2, lex_dict
    if aug1_list is not None and aug2_list is not None:
        aug1 = vocab_to_dict(vocab, vecs, aug1_list)
        aug2 = vocab_to_dict(vocab, vecs, aug2_list)
        return t1, t2, a1, a2, aug1, aug2
    return t1, t2, a1, a2


def dict_to_vocab_vecs(dictionary):
    vocab = {}
    vecs = []
    counter = 0
    for word in dictionary:
        vocab[word] = counter
        vecs.append(dictionary[word])
        counter += 1
    return vocab, vecs


# Computes a Principal Component Analysis (PCA) on up to four dictionaries
def principal_componant_analysis(vector_dict1, vector_dict2=None, vector_dict3=None, vector_dict4=None):
    logging.info("PCA: Principal composant analysis started")
    vector_dict1_copy = create_duplicates(vector_dict1)
    array_words = []
    array2d = []
    for word in vector_dict1_copy:
        array_words.append(word)
        array2d.append(list(vector_dict1_copy[word]))
    if vector_dict2 is not None:
        vector_dict2_copy = create_duplicates(vector_dict2)
        for word in vector_dict2_copy:
            array_words.append(word)
            array2d.append(list(vector_dict2_copy[word]))
    if vector_dict3 is not None:
        vector_dict3_copy = create_duplicates(vector_dict3)
        for word in vector_dict3_copy:
            array_words.append(word)
            array2d.append(list(vector_dict3_copy[word]))
    if vector_dict4 is not None:
        vector_dict4_copy = create_duplicates(vector_dict4)
        for word in vector_dict4_copy:
            array_words.append(word)
            array2d.append(list(vector_dict4_copy[word]))
    pca2 = PCA(n_components=2)
    prinicpal_components = pca2.fit_transform(numpy.array(array2d))
    results = {}
    for i in range(len(array_words)):
        results[array_words[i]] = prinicpal_components[i]
    logging.info("PCA: Prinicipal composant analysis completed")
    return results


def principal_componant_analysis2(vecs):
    pca2 = PCA(n_components=2)
    prinicpal_components = pca2.fit_transform(numpy.array(vecs))
    return prinicpal_components
