import numpy
import psycopg2
import logging
import datetime

import calculation

# Has to be adjusted depending on server setup
database_user = 'postgres'
database_host = 'localhost'
database_password = ''


# Retrieved the word vector representation out of a database
def get_vector_from_database(word, database):
    # t1 = datetime.datetime.now()
    # print(t1)
    conn = None
    vector_dict = {}
    command = """
    SELECT vector FROM fasttext WHERE word = '{}'
    """.format(word)
    # print(command)
    try:
        conn = psycopg2.connect(dbname=database, user=database_user, host=database_host, password=database_password)
        cur = conn.cursor()
        cur.execute(command)
        logging.info("DB: Connected successfully to " + database)
        records = cur.fetchall()
        data = str(records)[3:-4]
        # print(data)
        vector_dict[word] = map(float, data.split('  '))
        logging.info("DB: Found vector for word: " + word)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
    # t2 = datetime.datetime.now()
    # print(t2)
    # print(t2-t1)
    return vector_dict


# Retrieves multiple word vectors of word lists out of the databases
def get_multiple_vectors_from_db(word_list, database):
    conn = None
    # integer = 0
    vector_dict = {}
    tablename = database

    try:
        conn = psycopg2.connect(dbname=database, user=database_user, host=database_host, password=database_password)
        cur = conn.cursor()
        logging.info("DB: Connected successfully to " + database)
        for word in word_list:
            command = """
                               SELECT vector FROM {} WHERE word = '{}'
                               """.format(tablename, word)
            try:

                cur.execute(command)
                # print(command)
                records = cur.fetchall()
                data = records[0]

                tokens = str(data)[2:-3].split('  ')
                vector = []
                for i in range(len(tokens)):
                    vector.append(float(tokens[i]))
                vector_dict[word] = map(float, vector)
                logging.info("DB: Found vector for " + word)
            except:
                logging.info("DB: No vector found for " + word)
                # print(command)
                pass
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error("DB: Database error", error)
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return vector_dict


# Retrieves four augmentations for a word from the database
def get_augmentation_from_db(word):
    conn = None
    augmentation = []
    try:
        conn = psycopg2.connect(dbname='augmentation', user=database_user, host=database_host, password=database_password)
        cur = conn.cursor()
        logging.info("DB: Connected successfully to augmentation")
        print("DB: Connected successfully to augmentation")
        command = """SELECT augment1, augment2, augment3, augment4 FROM augmentation WHERE word = '{}'""".format(word)
        cur.execute(command)
        records = cur.fetchall()
        augmentation = records[0]
        logging.info("DB: Found augmentation for " + word + ": " + str(augmentation))
        print("DB: Found augmentation for " + word + ": " + str(augmentation))
    except psycopg2.DatabaseError as error:
        logging.error("DB: Database error", error)
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return augmentation


# Retrieves four augmentations for each word of a word list
def get_multiple_augmentation_from_db(word_list, database):
    conn = None
    augmentations = {}
    try:
        conn = psycopg2.connect(dbname='augmentation', user=database_user, host=database_host, password=database_password)
        cur = conn.cursor()
        logging.info("DB: Connected successfully to augmentation")
        for word in word_list:
            try:
                command = """SELECT augment1, augment2, augment3, augment4 FROM augmentation WHERE word = '{}'""".format(word)
                cur.execute(command)
                records = cur.fetchall()
                data = records[0]
                augmentations[word] = data
                logging.info("DB: Found augmentation for " + word + ": " + str(data))
            except:
                logging.info("DB: No vector found for " + word)
                # data = augmentation.load_augment(word, da)
                data = get_vector_from_database(word, database)
                # augmentations[word] = data
                pass
    except psycopg2.DatabaseError as error:
        logging.error("DB: Database error", error)
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return augmentations


# In Development -- Needed for word analogy computing
# Retrieves vectors from first 10k database entries
def word_for_nearest_vector(request_vector, database):
    print(datetime.datetime.now())
    conn = None
    running = 0
    target_word = ''
    maximum_vector = []
    maximum_cosine = 0.0
    try:
        conn = psycopg2.connect(dbname=database, user=database_user, host=database_host, password=database_password)
        cur = conn.cursor()
        command = """SELECT vector FROM {} FETCH FIRST 10000 ONLY""".format(database)
        cur.execute(command)
        records = cur.fetchall()
        data = records[0]
        # cosine = []
        running += 1
        for string in data:
            running += 1
            tokens = str(string)[2:-3].split('  ')
            vector = []
            for i in range(len(tokens)):
                vector.append(float(tokens[i]))
            vector = numpy.array(vector)
            current_cosine = calculation.cosine_similarity(request_vector, vector)
            if current_cosine > maximum_cosine:
                maximum_vector = vector
                maximum_cosine = current_cosine
        command2 = """SELECT word FROM {} WHERE vector='{}'""".format(database, str(maximum_vector))
        cur.execute(command2)
        records2 = cur.fetchall()
        target_word = records2[0]
    except psycopg2.DatabaseError as error:
        logging.error("DB: Database error", error)
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return target_word
