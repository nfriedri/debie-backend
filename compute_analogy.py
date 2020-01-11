import numpy
import calculation
import database_handler

# In development


# Will return the result of a computed analogy
def full_analogy(word1, word2, word3):
    word1_copy, word2_copy, word3_copy = calculation.create_duplicates(word1, word2, word3)
    v1 = get_vector_from_small_dict(word1_copy)
    v2 = get_vector_from_small_dict(word2_copy)
    v3 = get_vector_from_small_dict(word3_copy)
    target_vector = numpy.array(v1) + numpy.array(v2) - numpy.array(v3)
    result_word = database_handler.word_for_nearest_vector(target_vector)
    return result_word


def get_vector_from_small_dict(dict1):
    vec = float
    for word in dict1:
        vec = list(dict1[word])
    return vec
