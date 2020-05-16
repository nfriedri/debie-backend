import calculation

uploaded_binary = ''

uploaded_filename = ''
uploaded_space = {}

uploaded_vectorfile = ''
uploaded_vocabfile = ''
uploaded_vecs = []
uploaded_vocab = {}


def get_vocab_vecs_from_upload():
    if uploaded_binary is 'true':
        return uploaded_vocab, uploaded_vecs
    else:
        return calculation.dict_to_vocab_vecs(uploaded_space)
