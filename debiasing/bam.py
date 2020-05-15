import numpy as np
import logging


def debias_proc(equality_sets, vocab, vecs):
    logging.info("Debi-Engine: BAM Debiasing started")
    A = []
    B = []
    vocab_list = []
    for eq in equality_sets:
        if eq[0] in vocab and eq[1] in vocab:
            A.append(vecs[vocab[eq[0]]])
            B.append(vecs[vocab[eq[1]]])
            vocab_list.append(eq[0])
            vocab_list.append(eq[1])
    A = np.array(A)
    B = np.array(B)

    product = np.matmul(A.transpose(), B)
    U, s, V = np.linalg.svd(product)
    # print(U.shape, V.shape)
    proj_mat = V  # np.matmul(U, V)
    res = np.matmul(vecs, proj_mat)
    logging.info("Debi-Engine: BAM Debiasing finished")
    return (res + vecs) / 2, proj_mat, vocab_list


'''
def bias_alignment_model(t1, t2, a1, a2, equality_sets):
    A = []
    B = []
    vocab_list = []
    for eq in equality_sets:
        if eq[0] in vocab and eq[1] in vocab:
            A.append(vecs[vocab[eq[0]]])
            B.append(vecs[vocab[eq[1]]])
            vocab_list.append(eq[0])
            vocab_list.append(eq[1])
    A = np.array(A)
    B = np.array(B)

    product = np.matmul(A.transpose(), B)
    U, s, V = np.linalg.svd(product)
    print(U.shape, V.shape)
    proj_mat = V  # np.matmul(U, V)
    res = np.matmul(vecs, proj_mat)
    return (res + vecs) / 2, proj_mat, vocab_list
'''
