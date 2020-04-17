import io


def load_dict_uploaded_file(filename):
    path = "uploads/" + filename
    fin = io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data
