import json


def dict_to_json(vector_dict):
    string_dict = {}
    for word in vector_dict:
        string_dict[word] = str(list(vector_dict[word]))
    return string_dict


def dict_keys_to_string(vector_dict):
    keys = ''
    for word in vector_dict.keys():
        keys += str(word) + ' '
    return keys


def json_vector_retrieval(vector_dict, not_found):
    found = dict_to_json(vector_dict)
    response = json.dumps({'Vector': found, 'NotFound': str(not_found)})
    return response


def json_to_bias_spec(content):
    target1, target2, attributes1, attributes2 = [], [], [], []
    if 'BiasSpecification' in content:
        target1 = content['BiasSpecification']['T1'].split(' ')
        target2 = content['BiasSpecification']['T2'].split(' ')
        attributes1 = content['BiasSpecification']['A1'].split(' ')
        attributes2 = content['BiasSpecification']['A2'].split(' ')
    else:
        target1 = content['T1'].split(' ')
        target2 = content['T2'].split(' ')
        attributes1 = content['A1'].split(' ')
        attributes2 = content['A2'].split(' ')
    return target1, target2, attributes1, attributes2


def bias_evaluation_json(scores, space, lower, t1, t2, a1, a2, not_found, deleted):
    # scores = dict_to_json(scores)
    t1 = dict_keys_to_string(t1)
    t2 = dict_keys_to_string(t2)
    a1 = dict_keys_to_string(a1)
    a2 = dict_keys_to_string(a2)
    response = json.dumps({"Scores": scores, "Space": space, "Lower": lower,
                           "BiasSpecification": {"T1": t1, "T2": t2, "A1": a1, "A2": a2,
                                                 "NotFound": not_found, "Deleted": deleted}})
    return response, 200
