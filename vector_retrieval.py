import data_controller


def retrieve_vector_single(vocab, vectors, target_word):
    found = {}
    if target_word in vocab:
        found[target_word] = vectors[vocab[target_word]]
    return found


def retrieve_vector_multiple(vocab, vectors, word_list):
    not_found = []
    found = {}
    for word in word_list:
        if word in vocab:
            found[word] = vectors[vocab[word]]
        else:
            not_found.append(word)
    return found, not_found


def retrieve_vector(number, uploaded, embeddings, target):
    if number is 'single':
        if embeddings is 'fasttext':
            return retrieve_vector_single(data_controller.fasttext_vocab, data_controller.fasttext_vectors, target)
        if embeddings is 'glove':
            return retrieve_vector_single(data_controller.glove_vocab, data_controller.glove_vectors, target)
        if embeddings is 'cbow':
            return retrieve_vector_single(data_controller.cbow_vocab, data_controller.cbow_vectors, target)
        if uploaded is 'true':
            return 'Solution needed'
        else:
            return 'error'
    if number is 'multiple':
        if embeddings is 'fasttext':
            return retrieve_vector_multiple(data_controller.fasttext_vocab, data_controller.fasttext_vectors, target)
        if embeddings is 'glove':
            return retrieve_vector_multiple(data_controller.glove_vocab, data_controller.glove_vectors, target)
        if embeddings is 'cbow':
            return retrieve_vector_multiple(data_controller.cbow_vocab, data_controller.cbow_vectors, target)
        if uploaded is 'true':
            return 'Solution needed'
        else:
            return 'error'
