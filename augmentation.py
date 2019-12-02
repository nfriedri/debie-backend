import io
import numpy
import database_handler
import calculation


def load_augment(word, sourcefile):
    fin = io.open(sourcefile, 'r', encoding='utf-8', newline='\n', errors='ignore')
    database = 'fasttext'
    n, d = map(int, fin.readline().split())
    word_vec = database_handler.get_vector_from_database(word, database)
    sourcevector = []
    for word in word_vec:
        sourcevector = numpy.array(list(word_vec[word]))
    cosinesim = {}
    integer = 0
    running = 0
    for line in fin:
        tokens = line.rstrip().split(' ')
        vector = []
        for i in range(1, len(tokens)):
            vector.append(float(tokens[i]))
        if tokens[0] != word:
            cosinesim[tokens[0]] = calculation.cosine_similarity(sourcevector, numpy.array(vector))
    maximum1 = max(cosinesim, key=lambda k: cosinesim[k])
    cosinesim.pop(maximum1)
    maximum2 = max(cosinesim, key=lambda k: cosinesim[k])
    cosinesim.pop(maximum2)
    maximum3 = max(cosinesim, key=lambda k: cosinesim[k])
    cosinesim.pop(maximum3)
    maximum4 = max(cosinesim, key=lambda k: cosinesim[k])
    cosinesim.pop(maximum4)
    return [maximum1, maximum2, maximum3, maximum4]


def load_multiple_augments(word_list, sourcefile):
    fin = io.open(sourcefile, 'r', encoding='utf-8', newline='\n', errors='ignore')
    database = 'fasttext'
    n, d = map(int, fin.readline().split())
    source_dict = database_handler.get_multiple_vectors_from_db(word_list, database)
    source_dict2 = calculation.create_duplicates(source_dict)
    source_vectors = []
    for word in source_dict2:
        source_vectors.append(numpy.array(list(source_dict2[word])))
    augmentations = {}
    for i in range(len(source_dict)):
        cosinesim = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            vector = []
            for i in range(1, len(tokens)):
                vector.append(float(tokens[i]))
            if tokens[0] != source_dict[i]:
                cosinesim[tokens[0]] = calculation.cosine_similarity(source_vectors[i], numpy.array(vector))
        maximum1 = max(cosinesim, key=lambda k: cosinesim[k])
        cosinesim.pop(maximum1)
        maximum2 = max(cosinesim, key=lambda k: cosinesim[k])
        cosinesim.pop(maximum2)
        maximum3 = max(cosinesim, key=lambda k: cosinesim[k])
        cosinesim.pop(maximum3)
        maximum4 = max(cosinesim, key=lambda k: cosinesim[k])
        cosinesim.pop(maximum4)
        augmentations[source_dict[i]] = [maximum1, maximum2, maximum3, maximum4]
    return augmentations


