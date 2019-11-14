import vectors
import database_handler

test_file = "C:\\Users\\Niklas\\Documents\\wiki-news-300d-1M.vec"


def get_vector_from_json(content):
    raw_t1 = content['T1']
    raw_t2 = content['T2']
    raw_a1 = content['A1']
    raw_a2 = content['A2']

    test_set1 = raw_t1.split(' ')
    test_set2 = raw_t2.split(' ')
    arg_set1 = raw_a1.split(' ')
    arg_set2 = raw_a2.split(' ')

    print("T1: " + str(len(test_set1)) + " T2: " + str(len(test_set2)) + " A1: " + str(len(arg_set1)) + " A2: " + str(
        len(arg_set2)))
    print(test_set1)
    print(test_set2)
    print(arg_set1)
    print(arg_set2)

    test_vectors1 = vectors.load_multiple_words(test_file, test_set1)
    print("vectors1 found")
    print(test_vectors1)

    print(len(test_vectors1))
    test_vectors2 = vectors.load_multiple_words(test_file, test_set2)
    print("vectors2 found")
    print(len(test_vectors2))
    arg_vectors1 = vectors.load_multiple_words(test_file, arg_set1)
    print("vectors3 found")
    print(len(arg_vectors1))
    arg_vectors2 = vectors.load_multiple_words(test_file, arg_set2)
    print("vectors4 found")
    print(len(arg_vectors2))

    return test_vectors1, test_vectors2, arg_vectors1, arg_vectors2


def get_vector_from_json_via_db(content):
    raw_t1 = content['T1']
    raw_t2 = content['T2']
    raw_a1 = content['A1']
    raw_a2 = content['A2']

    test_set1 = raw_t1.split(' ')
    test_set2 = raw_t2.split(' ')
    arg_set1 = raw_a1.split(' ')
    arg_set2 = raw_a2.split(' ')

    test_vectors1 = database_handler.get_multiple_vectors_from_db(test_set1)
    print("VECTORS1 found")

    test_vectors2 = database_handler.get_multiple_vectors_from_db(test_set2)
    print("VECTORS2 found")

    arg_vectors1 = database_handler.get_multiple_vectors_from_db(arg_set1)
    print("VECTORS3 found")

    arg_vectors2 = database_handler.get_multiple_vectors_from_db(arg_set2)
    print("VECTORS4 found")

    print()
    print('Sizes: ' + str(len(test_vectors1)) + " " + str(len(test_vectors2)) + str(len(arg_vectors1)) + str(len(test_vectors2)))

    return test_vectors1, test_vectors2, arg_vectors1, arg_vectors2
