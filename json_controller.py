from flask import jsonify


def json_vector_retrieval(vector_dict, not_found):
    if not_found is None:
        response = jsonify(Word=[word for word in vector_dict], Vector=[list(vector_dict[vec]) for vec in vector_dict])
    else:
        response = jsonify(Word=[word for word in vector_dict], Vector=[list(vector_dict[vec]) for vec in vector_dict], NotFound=[list(not_found)])


