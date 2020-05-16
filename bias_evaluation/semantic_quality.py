# Providing Simlex and Wordsim as implicit scores
import codecs
from data_controller import load_simlex, simlex_999, wordsim
from scipy.stats import stats
import numpy as np


def eval_simlex(vocab, vecs, sim_type):
    simlex = None
    if sim_type == 'SimLex':
        simlex = load_simlex(simlex_999)
    if sim_type == 'WordSim':
        simlex = load_simlex(wordsim)
    preds = []
    golds = []
    cnt = 0
    for s in simlex:
        if s[0] in vocab and s[1] in vocab:
            vec1 = vecs[vocab[s[0]]]
            vec2 = vecs[vocab[s[1]]]
            sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            preds.append(sim)
            golds.append(s[2])
        else:
            cnt += 1
    print("Didn't find " + str(cnt) + " pairs")
    pearson = stats.pearsonr(golds, preds)[0]
    spearman = stats.spearmanr(golds, preds)[0]
    return pearson, spearman
