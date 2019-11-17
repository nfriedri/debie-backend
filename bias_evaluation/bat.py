import calculation
import logging


# Method starts biased analogy test
def biased_analogy_test(test_set1, test_set2, argument_set1, argument_set2):
    logging.info("BAT: Calculation started:")
    test1, test2, argument1, argument2 = calculation.create_duplicates(test_set1, test_set2, argument_set1,
                                                                       argument_set2)
    numpy_test1 = calculation.create_numpy_vector(test1)
    numpy_test2 = calculation.create_numpy_vector(test2)
    numpy_arg1 = calculation.create_numpy_vector(argument1)
    numpy_arg2 = calculation.create_numpy_vector(argument2)
    logging.info("BAT: Vector dictionaries and lists prepared successfully")
    # Calculate query vectors for each combination of words
    query_vectors1, query_vectors2 = query_calculation(numpy_test1, numpy_test2, numpy_arg1, numpy_arg2)

    # Rank vectors after euclidean distance to query vectors
    rank_result = vector_ranking(query_vectors1, query_vectors2, numpy_arg1, numpy_arg2)
    logging.log("BAT: Finished calculation")
    logging.log("BAT: Results: " + str(rank_result))
    return rank_result


# Calculate query vectors for each combination of words
def query_calculation(numpy_test1, numpy_test2, numpy_arg1, numpy_arg2):
    query_vec1 = []
    query_vec2 = []
    logging.info("BAT: Started Query Calculation")
    integer = 0
    for i in range(len(numpy_test1)):
        for j in range(len(numpy_test2)):
            for k in range(len(numpy_arg1)):
                for l in range(len(numpy_arg2)):
                    query1 = numpy_test1[i] - numpy_test2[j] + numpy_arg2[l]
                    query2 = numpy_arg1[k] - numpy_test1[i] + numpy_test2[j]
                    query_vec1.append(query1)
                    query_vec2.append(query2)
                    integer += 1
                    if i == 1000:
                        print(integer)
    logging.info("BAT: Finished query calculation successfully")
    return query_vec1, query_vec2


# Rank vectors after euclidean distance to query vectors
def vector_ranking(query_vectors1, query_vectors2, arg_vectors1, arg_vectors2):
    logging.log("BAT: Started vector ranking")
    biased = 0
    others = 0
    print(len(query_vectors1) * len(arg_vectors2))
    for i in range(len(query_vectors1)):
        for j in range(len(arg_vectors2)):
            if calculation.euclidean_distance(query_vectors1[i], arg_vectors2[j]) > calculation.euclidean_distance(
                    query_vectors1[i], arg_vectors1[j]):
                biased += 1
                # if biased % 1000 == 0:
                   # print(biased)
            else:
                others += 1
    for i in range(len(query_vectors2)):
        for j in range(len(arg_vectors1)):
            if calculation.euclidean_distance(query_vectors2[i], arg_vectors1[j]) > calculation.euclidean_distance(
                    query_vectors2[i], arg_vectors2[j]):
                biased += 1
                print('BIASED')
            else:
                others += 1
    logging.log("BAT: Finished vector ranking")
    return biased / others
