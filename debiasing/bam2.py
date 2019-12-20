import numpy
import calculation


def bias_alignment_model(target_set1, target_set2, augments1, augments2):
    target1_copy, target2_copy = calculation.create_duplicates(target_set1, target_set2)
    augments1_copy, augments2_copy = calculation.create_duplicates(augments1, augments2)

    word_list = []
    aug1_list = []
    aug2_list = []
    vector_list = []

    for word in target1_copy:
        word_list.append(word)
        vector_list.append(list(target1_copy[word]))
    for word in target2_copy:
        word_list.append(word)
        vector_list.append(list(target2_copy[word]))
    for word in augments1_copy:
        element = numpy.array(list(augments1_copy[word]))
        aug1_list.append(element)
    for word in augments2_copy:
        element = numpy.array(list(augments2_copy[word]))
        aug2_list.append(element)

    term_pairs = []
    for i in range(len(aug1_list)):
        for j in range(len(aug2_list)):
            term_pairs.append([aug1_list[i], aug2_list[j]])

    x_t1 = numpy.array([term_pairs[i][0] for i in range(len(term_pairs))])
    x_t2 = numpy.array([term_pairs[i][1] for i in range(len(term_pairs))])
    multi = numpy.matmul(numpy.transpose(x_t1), x_t2)
    u, s, vh = numpy.linalg.svd(multi)
    w_matrix = numpy.matmul(u, vh)
    new_x_matrix = numpy.matmul(vector_list, w_matrix)
    result_matrix = 0.5 * (numpy.array(vector_list) + new_x_matrix)
    result_dict = {}
    for i in range(len(word_list)):
        result_dict[word_list[i]] = result_matrix[i]
    t1 = {}
    t2 = {}
    for word in result_dict:
        if word in target1_copy:
            t1[word] = result_dict[word]
        elif word in target2_copy:
            t2[word] = result_dict[word]

    return t1, t2




