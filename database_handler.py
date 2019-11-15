import psycopg2


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
        records = cur.fetchall()
        data = str(records)[3:-4]
        print(data)
        vector_dict[word] = map(float, data.split('  '))
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
    dbname = 'fasttext2'
    if database == 'fasttextdb':
        dbname = 'fasttext2'
    if database == 'skipgramdb':
        dbname = 'skipgram'
    if database == 'cbowdb':
        dbname = 'cbow'
    try:
        conn = psycopg2.connect(dbname=dbname, user='postgres', host='', password='audi')
        cur = conn.cursor()
        for word in word_list:
            try:
                command = """
                   SELECT vector FROM fasttext WHERE word = '{}'
                   """.format(word)
                cur.execute(command)
                records = cur.fetchall()
                data = records[0]

                tokens = str(data)[2:-3].split('  ')
                vector = []
                for i in range(len(tokens)):
                    vector.append(float(tokens[i]))
                vector_dict[word] = map(float, vector)
                print(word + ' Added to dict')
            except:
                print('Error')
                pass
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return vector_dict
