import numpy
import calculation


def bias_alignment_model(target_set1, target_set2):
    target1_copy, target2_copy = calculation.create_duplicates(target_set1, target_set2)
    target1, target2 = calculation.transform_multiple_dicts_to_lists(target1_copy, target2_copy)
    vector_space = target1 + target2
    word_space_list = list(target_set1) + list(target_set2)
    matrix_xt1 = stack_vectors(target1)
    matrix_xt2 = stack_vectors(target2)
    matrix_wx = orthogonal_map(matrix_xt1, matrix_xt2)
    debiased_space = bam_debiased_space(vector_space, matrix_wx)
    result_dict = {}
    for i in range(len(word_space_list)):
        result_dict[word_space_list[i]] = debiased_space[i]
    return result_dict


def stack_vectors(vector_list):
    stacked_vecs = []
    for i in range(1, len(vector_list)-1):
        vector = numpy.concatenate((numpy.array(vector_list[i-1]), numpy.array(vector_list[i+1])), axis=None)
        stacked_vecs.append(vector)
    stacked_vecs.append(numpy.concatenate((numpy.array(vector_list[0]), numpy.array(vector_list[len(vector_list)-1])), axis=None))
    stacked_vecs.append(numpy.concatenate((numpy.array(vector_list[len(vector_list)-1]), numpy.array(vector_list[0])), axis=None))
    return stacked_vecs


def orthogonal_map(xt1, xt2):
    u, s, vh = numpy.linalg.svd(numpy.matmul(xt2, numpy.transpose(xt1)))
    wx = numpy.matmul(u, vh)
    return wx


def bam_debiased_space(vector_space, wx):
    new_vec_space = (0.5 * numpy.array(vector_space) + numpy.matmul(numpy.array(vector_space), numpy.array(wx)))
    return new_vec_space
