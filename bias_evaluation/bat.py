import numpy as np


# Computes the Bias Analogy Test (BAT) on a bias specification
def bias_analogy_test(target1, target2, attribute1, attribute2):
    # target1, target2, attribute1, attribute2 = calculation.create_duplicates(target_set1, target_set2, attribute_set1, attribute_set2)
    counter = 0
    vocab = {}
    vectors = []
    target_1 = []
    target_2 = []
    attributes_1 = []
    attributes_2 = []

    for word in target1:
        vocab[word] = counter
        counter += 1
        target_1.append(word)
        vectors.append(np.array(list(target1[word])))
    for word in target2:
        vocab[word] = counter
        counter += 1
        target_2.append(word)
        vectors.append(np.array(list(target2[word])))
    for word in attribute1:
        vocab[word] = counter
        counter += 1
        attributes_1.append(word)
        vectors.append(np.array(list(attribute1[word])))
    for word in attribute2:
        vocab[word] = counter
        counter += 1
        attributes_2.append(word)
        vectors.append(np.array(list(attribute2[word])))
    # print(len(vectors))
    # print(vectors[0])

    attributes_paired = []
    for a1 in attributes_1:
        for a2 in attributes_2:
            attributes_paired.append((a1, a2))
    # print(len(attributes_paired))
    # print(len(attributes_paired[0]))
    # print(attributes_paired)

    temporary_vocab = list(set(target_1 + target_2 + attributes_1 + attributes_2))
    # print(temporary_vocab)
    dictionary_list = []
    vector_matrix = []
    for w in temporary_vocab:
        if w in vocab:
            vector_matrix.append(vectors[vocab[w]])
            dictionary_list.append(w)

    vector_matrix = np.array(vector_matrix)
    # print(vector_matrix[10])
    vocab = {dictionary_list[i]: i for i in range(len(dictionary_list))}

    eq_pairs = []
    for t1 in target_1:
        for t2 in target_2:
            eq_pairs.append((t1, t2))
    # print(len(eq_pairs))
    # print(eq_pairs)

    for pair in eq_pairs:
        t1 = pair[0]
        t2 = pair[1]
        # print(t1)
        # print(t2)
        vec_t1 = vector_matrix[vocab[t1]]
        vec_t2 = vector_matrix[vocab[t2]]
        # print(len(vec_t1))
        biased = []
        totals = []
        for a1, a2 in attributes_paired:
            vectors_a1 = vector_matrix[vocab[a1]]
            vectors_a2 = vector_matrix[vocab[a2]]

            diff_vec = vec_t1 - vec_t2

            query_1 = diff_vec + vectors_a2
            query_2 = vectors_a1 - diff_vec

            sims_q1 = np.sum(np.square(vector_matrix - query_1), axis=1)
            sorted_q1 = np.argsort(sims_q1)
            indices = np.where(sorted_q1 == vocab[a1])[0][0]
            other_attr_2 = [x for x in attributes_2 if x != a2]
            indices_other = [np.where(sorted_q1 == vocab[x])[0][0] for x in other_attr_2]
            number_biased = [x for x in indices_other if indices < x]
            biased.append(len(number_biased))
            totals.append(len(indices_other))

            sims_q2 = np.sum(np.square(vector_matrix - query_2), axis=1)
            sorted_q2 = np.argsort(sims_q2)
            indices = np.where(sorted_q2 == vocab[a2])[0][0]
            other_attr_1 = [x for x in attributes_1 if x != a1]
            indices_other = [np.where(sorted_q2 == vocab[x])[0][0] for x in other_attr_1]
            number_biased = [x for x in indices_other if indices < x]
            biased.append(len(number_biased))
            totals.append(len(indices_other))

    result = sum(biased) / sum(totals)
    return result
