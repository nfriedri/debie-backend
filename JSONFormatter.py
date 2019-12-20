import calculation
import vectors
import database_handler
import augmentation
import logging


def get_json_vector_from_file(content):
    logging.info("DB: Vector retrieval started")
    logging.info("DB: Searching for following vectors:")
    logging.info("DB:" + str(content))
    raw_t1 = content['T1'].split(' ')
    raw_t2 = content['T2'].split(' ')
    raw_a1 = content['A1'].split(' ')
    raw_a2 = content['A2'].split(' ')
    logging.info("T1: " + str(len(raw_t1)) + " T2: " + str(len(raw_t2)) + " A1: " + str(len(raw_a1)) + " A2: " + str(
        len(raw_a2)))

    test_vectors1 = vectors.load_multiple_words(fasttext, raw_t1)
    logging.info("vectors1 found")
    test_vectors2 = vectors.load_multiple_words(fasttext, raw_t2)
    logging.info("vectors2 found")
    arg_vectors1 = vectors.load_multiple_words(fasttext, raw_a1)
    logging.info("vectors3 found")
    arg_vectors2 = vectors.load_multiple_words(fasttext, raw_a2)
    logging.info("vectors4 found")
    logging.info("Retrieved set sizes: " + str(len(test_vectors1)) + " " + str(len(test_vectors2)) + " " + str(
        len(arg_vectors1)) + " " + str(len(arg_vectors2)))

    return test_vectors1, test_vectors2, arg_vectors1, arg_vectors2


def retrieve_vector_from_db(content):
    logging.info("DB: Retrieval of single vector started")
    logging.info("DB: Searching for following vector:")
    logging.info("DB:" + str(content))
    database = content['EmbeddingSpace']
    raw_data = content['data'].rsplit()
    vector = database_handler.get_vector_from_database(raw_data, database)
    logging.info("DB: Found vector to word " + raw_data)
    return vector


def retrieve_vectors_evaluation(content, database):
    logging.info("DB: Retrieval of multiple vectors started")
    logging.info("DB: Searching for following vectors:")
    logging.info("DB:" + str(content))
    raw_t1 = content['T1'].split(' ')
    raw_t2 = content['T2'].split(' ')
    raw_a1 = content['A1'].split(' ')
    raw_a2 = content['A2'].split(' ')
    if database is None:
        database = 'fasttext'
    if database == "fasttext" or database == "skipgram" or database == "cbow" or database == "glove":
        test_vectors1 = database_handler.get_multiple_vectors_from_db(raw_t1, database)
        logging.info("DB: First set added to memory")
        test_vectors2 = database_handler.get_multiple_vectors_from_db(raw_t2, database)
        logging.info("DB: Second set added to memory")
        arg_vectors1 = database_handler.get_multiple_vectors_from_db(raw_a1, database)
        logging.info("DB: Third set added to memory")
        arg_vectors2 = database_handler.get_multiple_vectors_from_db(raw_a2, database)
        logging.info("DB: Fourth set added to memory")
        logging.info("DB: Found set sizes: " + str(len(test_vectors1)) + " " + str(len(test_vectors2)) + " " + str(
            len(arg_vectors1)) + " " + str(len(test_vectors2)))
    else:
        file = "uploads\\files\\" + database
        test_vectors1 = vectors.load_multiple_words(file, raw_t1)
        logging.info("DB: First set added to memory")
        test_vectors2 = vectors.load_multiple_words(file, raw_t2)
        logging.info("DB: Second set added to memory")
        arg_vectors1 = vectors.load_multiple_words(file, raw_a1)
        logging.info("DB: Third set added to memory")
        arg_vectors2 = vectors.load_multiple_words(file, raw_a2)

    return test_vectors1, test_vectors2, arg_vectors1, arg_vectors2


def retrieve_vectors_debiasing(content, database, augment_flag=True):
    logging.info("DB: Retrieval of multiple vectors started")
    logging.info("DB: Searching for following vectors:")
    logging.info("DB:" + str(content))
    raw_t1 = content['T1'].split(' ')
    raw_t2 = content['T2'].split(' ')
    if database is None:
        database = 'fasttext'
    if database == "uploadSpace":
        # TODO Somehow retrieve filename from last upload
        file = "uploads\\files\\" + database
        print(file)
        test_vectors1 = vectors.load_multiple_words(file, raw_t1)
        logging.info("DB: First set added to memory")
        test_vectors2 = vectors.load_multiple_words(file, raw_t2)
        logging.info("DB: Second set added to memory")
        if augment_flag:
            raw_aug1 = content['A1'].split(' ')
            raw_aug2 = content['A2'].split(' ')
            aug_target1 = vectors.load_multiple_words(file, raw_aug1)
            aug_target2 = vectors.load_multiple_words(file, raw_aug2)
        else:
            aug_target1 = augmentation.load_multiple_augments(raw_t1, file)
            aug_target2 = augmentation.load_multiple_augments(raw_t2, file)
    else:
        test_vectors1 = database_handler.get_multiple_vectors_from_db(raw_t1, database)
        logging.info("DB: First set added to memory")
        test_vectors2 = database_handler.get_multiple_vectors_from_db(raw_t2, database)
        logging.info("DB: Second set added to memory")
        logging.info("DB: Found set sizes: " + str(len(test_vectors1)) + " " + str(len(test_vectors2)))
        if augment_flag:
            raw_aug1 = content['A1'].split(' ')
            raw_aug2 = content['A2'].split(' ')
            aug_target1 = database_handler.get_multiple_vectors_from_db(raw_aug1, database)
            aug_target2 = database_handler.get_multiple_vectors_from_db(raw_aug2, database)
        else:
            aug_target1 = augmentation.load_multiple_augments(raw_t1, database)
            aug_target2 = augmentation.load_multiple_augments(raw_t2, database)

    return test_vectors1, test_vectors2, aug_target1, aug_target2


def retrieve_vectors_from_json(content):
    logging.info("DB: Loading Vectors from JSON Input.")
    target_dict1 = {}
    target_dict2 = {}
    attribute_dict1 = {}
    attribute_dict2 = {}
    for pair in content['T1']:
        word = pair['word']
        vec = pair['vec']
        target_dict1[word] = vec.split(' ')
    for pair in content['T2']:
        word = pair['word']
        vec = pair['vec']
        target_dict2[word] = vec.split(' ')
    for pair in content['A1']:
        word = pair['word']
        vec = pair['vec']
        attribute_dict1[word] = vec.split(' ')
    for pair in content['A2']:
        word = pair['word']
        vec = pair['vec']
        attribute_dict2[word] = vec.split(' ')
    logging.info("DB: Loaded JSON Input successfully.")
    return target_dict1, target_dict2, attribute_dict1, attribute_dict2


def dict_to_json(vector_dict):
    vector_dict_copy = calculation.create_duplicates(vector_dict)
    string_dict = {}
    for word in vector_dict_copy:
        string_dict[word] = str(list(vector_dict_copy[word]))
    return string_dict
