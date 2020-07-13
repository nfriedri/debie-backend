import codecs
import pickle
import io
import numpy as np

import upload_controller

fasttext_200k_vocab = 'data/ft.wiki.en.300.vocab'
fasttext_200k_vectors = 'data/ft.wiki.en.300.vectors'
glove_200k_vocab = 'data/glove_200k.vocab'
glove_200k_vectors = 'data/glove_200k.vec'
cbow_200k_vocab = 'data/w2v_cbow_200k.vocab'
cbow_200k_vectors = 'data/w2v_cbow_200k.vec'
augmentations_postspec = 'data/augmentations_postspec.vocab'
simlex_999_path = 'data/simlex999/SimLex-999.txt'
wordsim_path = 'data/wsim353/wsim.txt'


def load_vocab_binary(path, inverse=False):
    vocab = pickle.load(open(path, "rb"))
    if inverse:
        vocab_inv = {v: k for k, v in vocab.items()}
        return vocab, vocab_inv
    else:
        return vocab


def load_vectors_binary(path, normalize=False):
    vecs = np.load(path, allow_pickle=True)
    if normalize:
        vecs_norm = vecs / np.transpose([np.linalg.norm(vecs, 2, 1)])
        return vecs, vecs_norm
    else:
        return vecs


def load_binary_embeddings(vocab_path, vecs_path, inverse=False, normalize=False):
    vocab = load_vocab_binary(vocab_path, inverse)
    vecs = load_vectors_binary(vecs_path, normalize)
    return vocab, vecs


def load_augmentations(augmentations_path):
    with open(augmentations_path, 'rb') as handle:
        augmentations = pickle.load(handle)
        return augmentations


def load_dict_uploaded_file(filename):
    path = "uploads/" + filename
    fin = io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        ln = np.array(tokens[1:])
        data[tokens[0]] = ln.astype(np.float)
    upload_controller.uploaded_filename = filename
    upload_controller.uploaded_space = data
    return 'SEE CONSOLE'


def load_binary_uploads(vocab_filename, vecs_filename):
    path_vocab = "uploads/" + vocab_filename
    path_vecs = "uploads/" + vecs_filename
    vocab, vecs = load_binary_embeddings(path_vocab, path_vecs)
    upload_controller.uploaded_vocabfile = vocab_filename
    upload_controller.uploaded_vectorfile = vecs_filename
    upload_controller.uploaded_vocab = vocab
    upload_controller.uploaded_vecs = vecs
    return 'SEE CONSOLE'


def load_simlex(path):
    simlex_data = [line.strip() for line in list(codecs.open(path, "r", encoding='utf8', errors='replace').readlines())]
    # print(simlex_data)
    simlex = [(line.split("\t")[0].lower(), line.split("\t")[1].lower(), float(line.split("\t")[3])) for line in simlex_data]
    # print(path)
    # print(simlex)
    return simlex


def load_wordsim(path):
    wordsim_data = [line.strip() for line in list(codecs.open(path, "r", encoding='utf8', errors='replace').readlines())]
    wordsim = [(line.split("\t")[1].lower(), line.split("\t")[2].lower(), float(line.split("\t")[3])) for line in wordsim_data]
    return wordsim


def load_lex_by_start():
    simlex_data = load_simlex(simlex_999_path)
    wordsim_data = load_wordsim(wordsim_path)
    simlex_vocab, wordsim_vocab = [], []
    for s in simlex_data:
        simlex_vocab.append(s[0])
        simlex_vocab.append(s[1])
    for w in wordsim_data:
        wordsim_vocab.append(w[0])
        wordsim_vocab.append(w[1])
    return simlex_vocab, simlex_data, wordsim_vocab, wordsim_data


def load_embeddings_by_start():
    fasttext_vocab, fasttext_vectors = load_binary_embeddings(fasttext_200k_vocab, fasttext_200k_vectors, inverse=False,
                                                              normalize=False)
    glove_vocab, glove_vectors = load_binary_embeddings(glove_200k_vocab, glove_200k_vectors, inverse=False,
                                                        normalize=False)
    cbow_vocab, cbow_vectors = load_binary_embeddings(cbow_200k_vocab, cbow_200k_vectors, inverse=False,
                                                      normalize=False)
    return fasttext_vocab, fasttext_vectors, glove_vocab, glove_vectors, cbow_vocab, cbow_vectors


fasttext_vocab, fasttext_vectors, glove_vocab, glove_vectors, cbow_vocab, cbow_vectors = load_embeddings_by_start()
simlex_vocab, simlex_data, wordsim_vocab, wordsim_data = load_lex_by_start()
augmentations = load_augmentations(augmentations_postspec)
print('Data_Handler started')
