import calculation


def simlex_benchmark(test_set1, test_set2, argument_set1, argument_set2):
    test1, test2, arg1, arg2 = calculation.create_duplicates_four(test_set1, test_set2, argument_set1, argument_set2)
    test1, test2, arg1, arg2 = calculation.transform_multiple_dicts_to_lists(test1, test2, arg1, arg2)
    vector_list = test1 + test2 + arg1 + arg2

    predictions = []
    gold_standard = []
    counter = 0
    for vector in vector_list:
        similarity = calculation.cosine_similarity()
    return 0
