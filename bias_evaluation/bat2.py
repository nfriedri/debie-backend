import numpy as np

import calculation


def bias_analogy_test(target___1, target___2, attr___1, attr___2):
    tar1, tar2, att1, att2 = calculation.create_duplicates(target___1, target___2, attr___1, attr___2)
    counter = 0
    vocab = {}
    vecs = []
    target_1 = []
    target_2 = []
    attributes_1 = []
    attributes_2 = []
    for word in tar1:
        target_1.append(word)
        vecs.append(np.array(list(tar1[word])))
        vocab[word] = counter
        counter += 1
    for word in tar2:
        vocab[word] = counter
        counter += 1
        target_2.append(word)
        vecs.append(np.array(list(tar2[word])))
    for word in att1:
        vocab[word] = counter
        counter += 1
        attributes_1.append(word)
        vecs.append(np.array(list(att1[word])))
    for word in att2:
        vocab[word] = counter
        counter += 1
        attributes_2.append(word)
        vecs.append(np.array(list(att2[word])))
    atts_paired = []
    for a1 in attributes_1:
        for a2 in attributes_2:
            atts_paired.append((a1, a2))

    tmp_vocab = list(set(target_1 + target_2 + attributes_1 + attributes_2))
    dicto = []
    matrix = []
    for w in tmp_vocab:
        if w in vocab:
            matrix.append(vecs[vocab[w]])
            dicto.append(w)

    vecs = np.array(matrix)
    vocab = {dicto[i]: i for i in range(len(dicto))}
    eq_pairs = []
    for t1 in target_1:
        for t2 in target_2:
            eq_pairs.append((t1, t2))

    for pair in eq_pairs:
        t1 = pair[0]
        t2 = pair[1]
        vec_t1 = vecs[vocab[t1]]
        vec_t2 = vecs[vocab[t2]]

        biased = []
        totals = []
        for a1, a2 in atts_paired:
            vec_a1 = vecs[vocab[a1]]
            vec_a2 = vecs[vocab[a2]]

            diff_vec = vec_t1 - vec_t2

            query_1 = diff_vec + vec_a2
            query_2 = vec_a1 - diff_vec

            sims_q1 = np.sum(np.square(vecs - query_1), axis=1)
            sorted_q1 = np.argsort(sims_q1)
            ind = np.where(sorted_q1 == vocab[a1])[0][0]
            other_att_2 = [x for x in attributes_2 if x != a2]
            indices_other = [np.where(sorted_q1 == vocab[x])[0][0] for x in other_att_2]
            num_bias = [x for x in indices_other if ind < x]
            biased.append(len(num_bias))
            totals.append(len(indices_other))

            sims_q2 = np.sum(np.square(vecs - query_2), axis=1)
            sorted_q2 = np.argsort(sims_q2)
            ind = np.where(sorted_q2 == vocab[a2])[0][0]
            other_att_1 = [x for x in attributes_1 if x != a1]
            indices_other = [np.where(sorted_q2 == vocab[x])[0][0] for x in other_att_1]
            num_bias = [x for x in indices_other if ind < x]
            biased.append(len(num_bias))
            totals.append(len(indices_other))

        return sum(biased) / sum(totals)
