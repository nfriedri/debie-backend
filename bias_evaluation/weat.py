import numpy
import calculation
import logging


# TODO: Make accuracy optional
def word_embedding_association_test(target_set1, target_set2, argument_set1, argument_set2, accuracy=100):
    logging.info("WEAT: Started calculation")
    target1, target2, arg1, arg2 = calculation.create_duplicates(target_set1, target_set2, argument_set1,
                                                                 argument_set2)
    target1, target2, arg1, arg2 = calculation.transform_multiple_dicts_to_lists(target1, target2, arg1, arg2)
    target_list = target1 + target2
    logging.info("WEAT: Vector dictionaries and lists prepared successfully")
    # Calculate effect size
    effect_size = effect_size_calculation(target_list, target1, target2, arg1, arg2)
    # Calculate p_value
    logging.info("WEAT: Started p-value calculation")
    s_b_e = differential_association(target1, target2, arg1, arg2)
    s_b_e_all = sum_up_diff_ass_all_permutations(target_list, arg1, arg2, accuracy)
    p_value = p_value_calculation(s_b_e, s_b_e_all)
    logging.info("WEAT: Finished p-value calculation with result " + str(p_value))
    logging.info("WEAT: Finished calculation")
    logging.info("WEAT: Results: effect-size: " + str(effect_size) + " p-value: " + str(p_value))
    return effect_size, p_value


def differential_association(target_list1, target_list2, argument_list1, argument_list2):
    minuend = []
    subtrahend = []
    for target in target_list1:
        minuend.append(association(target, argument_list1, argument_list2))
    for target in target_list2:
        subtrahend.append(association(target, argument_list1, argument_list2))
    result = sum(minuend) - sum(subtrahend)
    return result


def association(target_word, argument_list1, argument_list2):
    minuend = []
    subtrahend = []
    for arg in argument_list1:
        minuend.append(calculation.cosine_similarity(target_word, arg))
    for arg in argument_list2:
        subtrahend.append(calculation.cosine_similarity(target_word, arg))
    minuend = sum(minuend) / len(argument_list1)
    subtrahend = sum(subtrahend) / len(argument_list2)
    return minuend - subtrahend


def random_permutation(test_list):
    half = int(len(test_list) / 2)
    permutation_list = numpy.random.permutation(test_list)
    return permutation_list[half:], permutation_list[:half]


def sum_up_diff_ass_all_permutations(test_list, argument_list1, argument_list2, accuracy):
    sum_up = []
    for i in range(accuracy):
        test1, test2 = random_permutation(test_list)
        sum_up.append(differential_association(test1, test2, argument_list1, argument_list2))
    return sum_up


def p_value_calculation(s_b_e, s_b_e_all):
    is_bigger = 0
    for i in range(len(s_b_e_all)):
        if s_b_e_all[i] > s_b_e:
            is_bigger += 1
    return is_bigger / len(s_b_e_all)


def effect_size_calculation(target_all, target_list1, target_list2, argument_list1, argument_list2):
    logging.info("WEAT: Started effect size calculation")
    mean_association1 = numpy.mean([association(word, argument_list1, argument_list2) for word in target_list1])
    mean_association2 = numpy.mean([association(word, argument_list1, argument_list2) for word in target_list2])
    standard_deviation = numpy.std([association(word, argument_list1, argument_list2) for word in target_all])
    result = (mean_association1 - mean_association2) / standard_deviation
    logging.info("WEAT: Finished effect size calculation with result " + str(result))
    return result
