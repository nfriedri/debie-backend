import psycopg2
import logging


def get_vector_from_database(word, database):
    conn = None
    vector_dict = {}
    command = """
    SELECT vector FROM fasttext WHERE word = '{}'
    """.format(word)
    print(command)
    try:
        conn = psycopg2.connect(dbname='fasttext2', user='postgres', host='', password='audi')
        cur = conn.cursor()
        cur.execute(command)
        logging.info("DB: Connected successfully to " + database)
        records = cur.fetchall()
        data = str(records)[3:-4]
        print(data)
        vector_dict[word] = map(float, data.split('  '))
        logging.info("DB: Found vector for word: " + word)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return vector_dict


def get_multiple_vectors_from_db(word_list, database):
    conn = None
    integer = 0
    vector_dict = {}
    tablename = database

    try:
        conn = psycopg2.connect(dbname=database, user='postgres', host='', password='audi')
        cur = conn.cursor()
        logging.info("DB: Connected successfully to " + database)
        for word in word_list:
            try:
                command = """
                   SELECT vector FROM {} WHERE word = '{}'
                   """.format(tablename, word)
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
                pass
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error("DB: Database error", error)
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return vector_dict
