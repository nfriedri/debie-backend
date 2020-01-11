import calculation
import vectors
import database_handler
import augmentation
import logging

fasttext = ''


# Parses JSON-input and calls methods for vector retrieval from above specified file
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


# Parses JSON-input and calls methods for vector retrieval from databases
def retrieve_vector_from_db(content):
    logging.info("DB: Retrieval of single vector started")
    logging.info("DB: Searching for following vector:")
    logging.info("DB:" + str(content))
    database = content['EmbeddingSpace']
    raw_data = content['data'].rsplit()
    vector = database_handler.get_vector_from_database(raw_data, database)
    logging.info("DB: Found vector to word " + raw_data)
    return vector


# Parses JSON-input for bias evaluation and calls methods for vector retrieval from databases
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


# Parses JSON-input for debiasing and calls methods for vector retrieval from databases with optional augmentations
def retrieve_vectors_debiasing(content, database, augment_flag):
    logging.info("DB: Retrieval of multiple vectors started")
    logging.info("DB: Searching for following vectors:")
    logging.info("DB:" + str(content))
    raw_t1 = content['T1'].split(' ')
    raw_t2 = content['T2'].split(' ')
    raw_a1 = content['A1'].split(' ')
    raw_a2 = content['A2'].split(' ')
    if database is None:
        database = 'fasttext'

    if database == "uploadSpace":
        file = "uploads\\files\\" + database
        target_vectors1 = vectors.load_multiple_words(file, raw_t1)
        logging.info("DB: First set added to memory")
        target_vectors2 = vectors.load_multiple_words(file, raw_t2)
        logging.info("DB: Second set added to memory")
        attributes1 = vectors.load_multiple_words(file, raw_a1)
        logging.info("DB: Third set added to memory")
        attributes2 = vectors.load_multiple_words(file, raw_a2)
        logging.info("DB: Fourth set added to memory")
        if augment_flag == 'true':
            raw_aug1 = content['AugT1'].split(' ')
            raw_aug2 = content['AugT2'].split(' ')
            raw_aug3 = content['AugA1'].split(' ')
            raw_aug4 = content['AugA2'].split(' ')
            augments_T1 = vectors.load_multiple_words(file, raw_aug1)
            augments_T2 = vectors.load_multiple_words(file, raw_aug2)
            augments_A1 = vectors.load_multiple_words(file, raw_aug3)
            augments_A2 = vectors.load_multiple_words(file, raw_aug4)
            logging.info("DB: Retrieved augmentations")
        else:
            aug_list_T1 = database_handler.get_multiple_augmentation_from_db(list(target_vectors1.keys()), database)
            aug_list_T2 = database_handler.get_multiple_augmentation_from_db(list(target_vectors2.keys()), database)
            aug_list_A1 = database_handler.get_multiple_augmentation_from_db(list(attributes1.keys()), database)
            aug_list_A2 = database_handler.get_multiple_augmentation_from_db(list(attributes2.keys()), database)
            augments_T1 = database_handler.get_multiple_vectors_from_db(aug_list_T1, database)
            augments_T2 = database_handler.get_multiple_vectors_from_db(aug_list_T2, database)
            augments_A1 = database_handler.get_multiple_vectors_from_db(aug_list_A1, database)
            augments_A2 = database_handler.get_multiple_vectors_from_db(aug_list_A2, database)
            logging.info("DB: Retrieved augmentations")
    else:
        target_vectors1 = database_handler.get_multiple_vectors_from_db(raw_t1, database)
        logging.info("DB: First set added to memory")
        target_vectors2 = database_handler.get_multiple_vectors_from_db(raw_t2, database)
        logging.info("DB: Second set added to memory")
        attributes1 = database_handler.get_multiple_vectors_from_db(raw_a1, database)
        logging.info("DB: Third set added to memory")
        attributes2 = database_handler.get_multiple_vectors_from_db(raw_a2, database)
        logging.info("DB: Fourth set added to memory")
        logging.info("DB: Found set sizes: " + str(len(target_vectors1)) + " " + str(len(target_vectors2)))
        if augment_flag == 'true':
            raw_aug1 = content['AugT1'].split(' ')
            raw_aug2 = content['AugT2'].split(' ')
            raw_aug3 = content['AugA1'].split(' ')
            raw_aug4 = content['AugA2'].split(' ')
            augments_T1 = database_handler.get_multiple_vectors_from_db(raw_aug1, database)
            augments_T2 = database_handler.get_multiple_vectors_from_db(raw_aug2, database)
            augments_A1 = database_handler.get_multiple_vectors_from_db(raw_aug3, database)
            augments_A2 = database_handler.get_multiple_vectors_from_db(raw_aug4, database)
            logging.info("DB: Retrieved augmentations")

        else:
            aug_list_T1 = database_handler.get_multiple_augmentation_from_db(list(target_vectors1.keys()), database)
            aug_list_T2 = database_handler.get_multiple_augmentation_from_db(list(target_vectors2.keys()), database)
            aug_list_A1 = database_handler.get_multiple_augmentation_from_db(list(attributes1.keys()), database)
            aug_list_A2 = database_handler.get_multiple_augmentation_from_db(list(attributes2.keys()), database)
            augments_T1 = database_handler.get_multiple_vectors_from_db(aug_list_T1, database)
            augments_T2 = database_handler.get_multiple_vectors_from_db(aug_list_T2, database)
            augments_A1 = database_handler.get_multiple_vectors_from_db(aug_list_A1, database)
            augments_A2 = database_handler.get_multiple_vectors_from_db(aug_list_A2, database)
            logging.info("DB: Retrieved augmentations")

    return target_vectors1, target_vectors2, attributes1, attributes2, augments_T1, augments_T2, augments_A1, augments_A2


# Parses JSON-input for bias evaluation and calls methods for vector retrieval from the input JSON-file
def retrieve_vectors_from_json_evaluation(content):
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


# Parses JSON-input for debiasing and calls methods for vector retrieval from the input JSON-file
def retrieve_vectors_from_json_debiasing(content):
    logging.info("DB: Loading Vectors from JSON Input.")
    target_dict1 = {}
    target_dict2 = {}
    attribute_dict1 = {}
    attribute_dict2 = {}
    augmentation_dict1 = {}
    augmentation_dict2 = {}
    augmentation_dict3 = {}
    augmentation_dict4 = {}
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
    for pair in content['AugT1']:
        word = pair['word']
        vec = pair['vec']
        augmentation_dict1[word] = vec.split(' ')
    for pair in content['AugT2']:
        word = pair['word']
        vec = pair['vec']
        augmentation_dict1[word] = vec.split(' ')
    for pair in content['AugA1']:
        word = pair['word']
        vec = pair['vec']
        augmentation_dict1[word] = vec.split(' ')
    for pair in content['AugA2']:
        word = pair['word']
        vec = pair['vec']
        augmentation_dict1[word] = vec.split(' ')
    logging.info("DB: Loaded JSON Input successfully.")
    return target_dict1, target_dict2, attribute_dict1, attribute_dict2, augmentation_dict1, augmentation_dict2, augmentation_dict3, augmentation_dict4


# Transformes dictionaries into JSON-format
def dict_to_json(vector_dict):
    vector_dict_copy = calculation.create_duplicates(vector_dict)
    string_dict = {}
    for word in vector_dict_copy:
        string_dict[word] = str(list(vector_dict_copy[word]))
    return string_dict


# Transforms keys of a dictioanry into a string
def dict_keys_to_string(vector_dict):
    vector_dict_copy = calculation.create_duplicates(vector_dict)
    keys = ''
    for word in vector_dict_copy.keys():
        keys += str(word) + ' '
    return keys
