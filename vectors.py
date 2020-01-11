import io


# Loads vectors from a file into the memory
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


# Loads the vector for one word out of the file
def load_word(fname, word):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    # n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        if tokens[0] == word:
            data[tokens[0]] = map(float, tokens[1:])
            return data


# Loads the vectors for a word list into the memory
def load_word_list(fname, word_list):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    # n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        for word in word_list:
            tokens = line.rstrip().split(' ')
            if tokens[0] == word:
                data[tokens[0]] = map(float, tokens[1:])
                print(str(tokens[0]) + " added")
                break
    return data


# Loads the vectors for a word list into the memory
def load_multiple_words(fname, word_list):
    vectors = {}
    for word in word_list:
        try:
            data = load_word(fname, word)
            vectors[word] = map(float,data[word])
            print(word)
            print(vectors[word])
        except TypeError:
            print(word + " not in List")

    return vectors
