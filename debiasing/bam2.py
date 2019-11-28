import numpy
import calculation


def bias_alignment_model(target_set1, target_set2):
    target1_copy, target2_copy = calculation.create_duplicates(target_set1, target_set2)
    target1_copy2, target2_copy2 = calculation.create_duplicates(target_set1, target_set2)
    # target1, target2 = calculation.transform_multiple_dicts_to_lists(target1_copy, target2_copy)
    term_pairs = []
    for t1 in target1_copy:
        for t2 in target2_copy:
            term_pairs.append([numpy.array(list(target1_copy[t1])), numpy.array(list(target2_copy[t2]))])
    complete_matrix = []
    word_list = []
    for t1 in target1_copy2:
        word_list.append(target1_copy2[t1])
        complete_matrix.append(numpy.array(list(target1_copy2[t1])))
    for t2 in target2_copy2:
        word_list.append(target2_copy2[t2])
        complete_matrix.append(numpy.array(list(target2_copy2[t2])))

    x_t1 = numpy.array([term_pairs[i][0] for i in range(len(term_pairs))])
    x_t2 = numpy.array([term_pairs[i][1] for i in range(len(term_pairs))])
    print(x_t1)
    print()
    print(x_t2)
    multi = numpy.matmul(x_t2, numpy.transpose(x_t1))
    u, s, vh = numpy.linalg.svd(multi)
    w_matrix = numpy.matmul(u, vh)
    new_x_matrix = numpy.matmul(complete_matrix, w_matrix)
    result_matrix = 0.5 * (complete_matrix + new_x_matrix)
    result_dict = {}
    for i in range(len(word_list)):
        result_dict[word_list[i]] = result_matrix[i]

    return result_dict


def bias_alignment_model2(target_set1, target_set2):
    target1_copy, target2_copy = calculation.create_duplicates(target_set1, target_set2)
    target1_copy2, target2_copy2 = calculation.create_duplicates(target_set1, target_set2)

    word_list = []
    target1_list = []
    target2_list = []
    vector_list = []

    for word in target1_copy:
        word_list.append(word)
        element = numpy.array(list(target1_copy[word]))
        target1_list.append(element)
        vector_list.append(element)
    for word in target2_copy:
        word_list.append(word)
        element = numpy.array(list(target2_copy[word]))
        target2_list.append(element)
        vector_list.append(element)

    term_pairs = []
    for i in range(len(target1_list)):
        for j in range(len(target2_list)):
            term_pairs.append([target1_list[i], target2_list[j]])

    x_t1 = numpy.array([term_pairs[i][0] for i in range(len(term_pairs))])
    x_t2 = numpy.array([term_pairs[i][1] for i in range(len(term_pairs))])
    print(len(x_t1))
    print(len(x_t2[1]))
    print()
    multi = numpy.matmul(x_t2, numpy.transpose(x_t1))
    u, s, vh = numpy.linalg.svd(multi)
    print(len(u))
    print(len(u[1]))
    print(len(vh))
    print(len(vh[1]))
    print()
    w_matrix = numpy.matmul(u, vh)
    print(len(w_matrix))
    print(len(w_matrix[1]))
    print(len(vector_list))
    new_x_matrix = numpy.dot(vector_list, w_matrix)
    result_matrix = 0.5 * (vector_list + new_x_matrix)
    result_dict = {}
    for i in range(len(word_list)):
        result_dict[word_list[i]] = result_matrix[i]

    return result_dict




