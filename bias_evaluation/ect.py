from scipy.stats import spearmanr
import calculation


def embedding_coherence_test(test_set1, test_set2, arg_set):
    # Create duplicates
    test1, test2, argument = calculation.create_duplicates(test_set1, test_set2, arg_set)
    # Transform vector sets in lists
    test_list1 = calculation.transform_dict_to_list(test1)
    test_list2 = calculation.transform_dict_to_list(test2)
    arg_list = calculation.transform_dict_to_list(argument)
    # print("Test List Sizes:  "+str(len(test_list1)) + "  " + str(len(test_list2)))
    mean_target_vector1 = calculation.target_set_mean_vector(test_list1)
    mean_target_vector2 = calculation.target_set_mean_vector(test_list2)

    array_sim1 = []
    array_sim2 = []
    for i in range(len(arg_list)):
        memory = arg_list[i]
        cos_sim1 = calculation.cosine_similarity(mean_target_vector1, memory)
        array_sim1.append(cos_sim1)
        cos_sim2 = calculation.cosine_similarity(mean_target_vector2, memory)
        array_sim2.append(cos_sim2)
    value_array, p_value = spearmanr(array_sim1, array_sim2)
    return value_array, p_value
