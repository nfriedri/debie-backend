# Providing Simlex and Wordsim as implicit scores
import codecs
from data_controller import simlex_data, wordsim_data
from scipy.stats import stats
import numpy as np


def eval_simlex(vocab, vecs, sim_type):
    lex_data = None
    if sim_type == 'SimLex':
        lex_data = simlex_data
    if sim_type == 'WordSim':
        lex_data = wordsim_data
    preds = []
    golds = []
    cnt = 0
    for s in lex_data:
        if s[0] in vocab and s[1] in vocab:
            vec1 = vecs[vocab[s[0]]]
            vec2 = vecs[vocab[s[1]]]
            sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            preds.append(sim)
            golds.append(s[2])
        else:
            cnt += 1
    pearson = stats.pearsonr(golds, preds)[0]
    spearman = stats.spearmanr(golds, preds)[0]

    return pearson, spearman
