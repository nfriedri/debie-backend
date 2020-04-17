import pickle

import numpy as np

fasttext_200k_vocab = 'data/ft.wiki.en.300.vocab'
fasttext_200k_vectors = 'data/ft.wiki.en.300.vectors'
glove_200k_vocab = 'data/glove_200k.vocab'
glove_200k_vectors = 'data/glove_200k.vec'
cbow_200k_vocab = 'data/w2v_cbow_200k.vocab'
cbow_200k_vectors = 'data/w2v_cbow_200k.vec'


def load_vocab_binary(path):
    vocab = pickle.load(open(path, "rb"))
    return vocab


def load_vectors_binary(path):
    vecs = np.load(path, allow_pickle=True)
    return vecs


def load_binary_embeddings(vocab_path, vecs_path):
    vocab = load_vocab_binary(vocab_path)
    vecs = load_vectors_binary(vecs_path)
    return vocab, vecs


def load_embeddings_by_start():
    fasttext_vocab, fasttext_vectors = load_binary_embeddings(fasttext_200k_vocab, fasttext_200k_vectors)
    glove_vocab, glove_vectors = load_binary_embeddings(glove_200k_vocab, glove_200k_vectors)
    cbow_vocab, cbow_vectors = load_binary_embeddings(cbow_200k_vocab, cbow_200k_vectors)
    return fasttext_vocab, fasttext_200k_vectors, glove_vocab, glove_vectors, cbow_vocab, cbow_vectors


fasttext_vocab, fasttext_vectors, glove_vocab, glove_vectors, cbow_vocab, cbow_vectors = load_embeddings_by_start()
print('Data_Handler started')
