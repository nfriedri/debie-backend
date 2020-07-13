import logging
import calculation
import numpy


# Computes the Word Embedding Coherence Test (WEAT)
def word_embedding_association_test(target1, target2, attribute1, attribute2, accuracy=100):
    logging.info("Eval-Engine: WEAT started")
    # target1, target2, arg1, arg2 = calculation.create_duplicates(target_set1, target_set2, attribute_set1, attribute_set2)
    t1, t2, a1, a2 = calculation.transform_multiple_dicts_to_lists(target1, target2, attribute1, attribute2)
    target_list = numpy.concatenate((t1, t2))
    # Calculate effect size
    effect_size = effect_size_calculation(target_list, t1, t2, a1, a2)
    logging.info("Eval-Engine: WEAT effect-size: " + str(effect_size))
    # Calculate p_value
    s_b_e = differential_association(t1, t2, a1, a2)
    s_b_e_all = sum_up_diff_ass_all_permutations(target_list, a1, a2, accuracy)
    p_value = p_value_calculation(s_b_e, s_b_e_all)
    logging.info("Eval-Engine: WEAT p-value " + str(p_value))

    return effect_size, p_value


# Compute a differential association
def differential_association(target_list1, target_list2, attribute_list1, attribute_list2):
    minuend = []
    subtrahend = []
    for target in target_list1:
        minuend.append(association(target, attribute_list1, attribute_list2))
    for target in target_list2:
        subtrahend.append(association(target, attribute_list1, attribute_list2))
    result = sum(minuend) - sum(subtrahend)
    return result


# Compute an association
def association(target_word, attribute_list1, attribute_list2):
    minuend = []
    subtrahend = []
    for arg in attribute_list1:
        minuend.append(calculation.cosine_similarity(target_word, arg))
    for arg in attribute_list2:
        subtrahend.append(calculation.cosine_similarity(target_word, arg))
    minuend = sum(minuend) / len(attribute_list1)
    subtrahend = sum(subtrahend) / len(attribute_list2)
    return minuend - subtrahend


# Retrieve a reandom permutation
def random_permutation(test_list):
    half = int(len(test_list) / 2)
    permutation_list = numpy.random.permutation(test_list)
    return permutation_list[half:], permutation_list[:half]


# Sum up all differential associations of the permutations
def sum_up_diff_ass_all_permutations(test_list, attribute_list1, attribute_list2, accuracy):
    sum_up = []
    for i in range(accuracy):
        test1, test2 = random_permutation(test_list)
        sum_up.append(differential_association(test1, test2, attribute_list1, attribute_list2))
    return sum_up


# Compute the p-value
def p_value_calculation(s_b_e, s_b_e_all):
    is_bigger = 0
    for i in range(len(s_b_e_all)):
        if s_b_e_all[i] > s_b_e:
            is_bigger += 1
    return is_bigger / len(s_b_e_all)


# Compute the effect-size
def effect_size_calculation(target_all, target_list1, target_list2, attribute_list1, attribute_list2):
    # print("WEAT: Started effect size calculation")
    mean_association1 = numpy.mean([association(word, attribute_list1, attribute_list2) for word in target_list1])
    mean_association2 = numpy.mean([association(word, attribute_list1, attribute_list2) for word in target_list2])
    standard_deviation = numpy.std([association(word, attribute_list1, attribute_list2) for word in target_all])
    result = (mean_association1 - mean_association2) / standard_deviation
    # print("WEAT: Finished effect size calculation with result " + str(result))
    return result
