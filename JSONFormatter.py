import calculation
import vectors
import database_handler
import logging

fasttext = "C:\\Users\\Niklas\\Documents\\wiki-news-300d-1M.vec"


def get_vector_from_json(content):
    raw_t1 = content['T1'].split(' ')
    raw_t2 = content['T2'].split(' ')
    raw_a1 = content['A1'].split(' ')
    raw_a2 = content['A2'].split(' ')

    print("T1: " + str(len(raw_t1)) + " T2: " + str(len(raw_t2)) + " A1: " + str(len(raw_a1)) + " A2: " + str(
        len(raw_a2)))

    test_vectors1 = vectors.load_multiple_words(fasttext, raw_t1)
    print("vectors1 found")
    print(test_vectors1)

    print(len(test_vectors1))
    test_vectors2 = vectors.load_multiple_words(fasttext, raw_t2)
    print("vectors2 found")
    print(len(test_vectors2))
    arg_vectors1 = vectors.load_multiple_words(fasttext, raw_a1)
    print("vectors3 found")
    print(len(arg_vectors1))
    arg_vectors2 = vectors.load_multiple_words(fasttext, raw_a2)
    print("vectors4 found")
    print(len(arg_vectors2))

    return test_vectors1, test_vectors2, arg_vectors1, arg_vectors2


def get_multiple_vectors_from_json(content, database):
    raw_t1 = content['T1'].split(' ')
    raw_t2 = content['T2'].split(' ')
    raw_a1 = content['A1'].split(' ')
    raw_a2 = content['A2'].split(' ')

    test_vectors1 = database_handler.get_multiple_vectors_from_db(raw_t1, database)
    logging.info("DB: First set added to memory")
    test_vectors2 = database_handler.get_multiple_vectors_from_db(raw_t2, database)
    logging.info("DB: Second set added to memory")
    arg_vectors1 = database_handler.get_multiple_vectors_from_db(raw_a1, database)
    logging.info("DB: Third set added to memory")
    arg_vectors2 = database_handler.get_multiple_vectors_from_db(raw_a2, database)
    logging.info("DB: Fourth set added to memory")
    logging.info("DB: Found set sizes: " + str(len(test_vectors1)) + " " + str(len(test_vectors2)) + " " + str(len(arg_vectors1)) + " " + str(len(test_vectors2)))

    return test_vectors1, test_vectors2, arg_vectors1, arg_vectors2


def retrieve_vectors(content, database):
    logging.info("DB: Vector retrieval started")
    logging.info("DB: Searching for following vectors:")
    logging.info("DB:" + str(content))
    target1 = {}
    target2 = {}
    argument1 = {}
    argument2 = {}
    if database is None:
        target1, target2, argument1, argument2 = get_multiple_vectors_from_json(content, 'fasttextdb')
    if database == 'fasttext':
        target1, target2, argument1, argument2 = get_multiple_vectors_from_json(content, 'fasttextdb')
    if database == 'skipgram':
        target1, target2, argument1, argument2 = get_multiple_vectors_from_json(content, 'skipgramdb')
    if database == 'cbow':
        target1, target2, argument1, argument2 = get_multiple_vectors_from_json(content, 'cbowdb')
    if database == 'glove':
        target1, target2, argument1, argument2 = get_multiple_vectors_from_json(content, 'glovedb')
    return target1, target2, argument1, argument2


def dict_to_json(vector_dict):
    vector_dict_copy = calculation.create_duplicates(vector_dict)
    string_dict = {}
    for word in vector_dict_copy:
        string_dict[word] = str(list(vector_dict_copy[word]))
    return string_dict
