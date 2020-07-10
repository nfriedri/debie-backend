import json
import logging


def dict_to_json(vector_dict):
    string_dict = {}
    for word in vector_dict:
        string_dict[word] = vector_dict[word].tolist()
    return string_dict


def dict_keys_to_string(vector_dict):
    keys = ''
    for word in vector_dict.keys():
        keys += str(word) + ' '
    keys = keys[:len(keys)-1]
    return keys


def json_vector_retrieval(vector_dict, not_found):
    found = dict_to_json(vector_dict)
    response = json.dumps({'Vector': found, 'NotFound': str(not_found)})
    return response


def json_augmentation_retrieval(augments, not_postspec):
    response = json.dumps({'Augmentations': augments, 'NotPostspecialized': not_postspec})
    return response


def json_to_debias_spec(content):
    print('Here: JSON to debias spec')
    target1, target2, attributes1, attributes2, augments1, augments2 = [], [], [], [], [], []
    if 'BiasSpecification' in content:
        target1 = content['BiasSpecification']['T1'].split(' ')
        target2 = content['BiasSpecification']['T2'].split(' ')
        attributes1 = content['BiasSpecification']['A1'].split(' ')
        attributes2 = content['BiasSpecification']['A2'].split(' ')
        if 'Augmentations1' in content['BiasSpecification']:
            print('Augments 1 in content')
            augments1 = content['BiasSpecification']['Augmentations1'].split(' ')
        if 'Augmentations2' in content['BiasSpecification']:
            augments2 = content['BiasSpecification']['Augmentations2'].split(' ')
    else:
        target1 = content['T1'].split(' ')
        target2 = content['T2'].split(' ')
        attributes1 = content['A1'].split(' ')
        attributes2 = content['A2'].split(' ')
        if 'Augmentations1' in content:
            print('Augments 1 in content')
            augments1 = content['Augmentations1'].split(' ')
        if 'Augmentations2' in content:
            augments2 = content['Augmentations2'].split(' ')
    logging.info("JsonController: Found following bias spec: T1: " + str(target1) + "; T2: " + str(target2) + "; A1: " +
                 str(attributes1) + " ; A2: " + str(attributes2) + " ; Aug1: " + str(augments1) + " ; Aug2: " +
                 str(augments2))
    return target1, target2, attributes1, attributes2, augments1, augments2


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


def debiasing_json(space, lower, method, pca, aug1_list, aug2_list,
                   t1, t2, a1, a2,
                   t1_deb, t2_deb, a1_deb, a2_deb, not_found, deleted,
                   t1_pca_bias=None, t2_pca_bias=None, a1_pca_bias=None, a2_pca_bias=None,
                   t1_pca_deb=None, t2_pca_deb=None, a1_pca_deb=None, a2_pca_deb=None, lex_dict=None):

    t1 = dict_to_json(t1)
    t2 = dict_to_json(t2)
    a1 = dict_to_json(a1)
    a2 = dict_to_json(a2)
    t1_deb = dict_to_json(t1_deb)
    t2_deb = dict_to_json(t2_deb)
    a1_deb = dict_to_json(a1_deb)
    a2_deb = dict_to_json(a2_deb)

    if pca == 'false':
        if lex_dict is None:
            response = json.dumps(
                {"Space": space, "Model": method, "Lower": lower, "PCA": pca,
                 "UsedAugmentations": {"A1": aug1_list, "A2": aug2_list},
                 "BiasedSpace:": {"T1": t1, "T2": t2, "A1": a1, "A2": a2},
                 "DebiasedSpace": {"T1": t1_deb, "T2": t2_deb, "A1": a1_deb, "A2": a2_deb},
                 "NotFound": not_found, "Deleted": deleted})
        else:
            lex_dict = dict_to_json(lex_dict)
            response = json.dumps(
                {"Space": space, "Model": method, "Lower": lower, "PCA": pca,
                 "UsedAugmentations": {"A1": aug1_list, "A2": aug2_list},
                 "BiasedSpace:": {"T1": t1, "T2": t2, "A1": a1, "A2": a2},
                 "DebiasedSpace": {"T1": t1_deb, "T2": t2_deb, "A1": a1_deb, "A2": a2_deb},
                 "NotFound": not_found, "Deleted": deleted, "LexDictionary": lex_dict})
    else:
        t1_pca_bias = dict_to_json(t1_pca_bias)
        t2_pca_bias = dict_to_json(t2_pca_bias)
        a1_pca_bias = dict_to_json(a1_pca_bias)
        a2_pca_bias = dict_to_json(a2_pca_bias)
        t1_pca_deb = dict_to_json(t1_pca_deb)
        t2_pca_deb = dict_to_json(t2_pca_deb)
        a1_pca_deb = dict_to_json(a1_pca_deb)
        a2_pca_deb = dict_to_json(a2_pca_deb)
        if lex_dict is None:
            response = json.dumps(
                {"Space": space, "Model": method, "Lower": lower, "PCA": pca,
                 "UsedAugmentations": {"A1": aug1_list, "A2": aug2_list},
                 "BiasedSpace:": {"T1": t1, "T2": t2, "A1": a1, "A2": a2},
                 "DebiasedSpace": {"T1": t1_deb, "T2": t2_deb, "A1": a1_deb, "A2": a2_deb},
                 "BiasedSpacePCA": {"T1": t1_pca_bias, "T2": t2_pca_bias, "A1": a1_pca_bias, "A2": a2_pca_bias},
                 "DebiasedSpacePCA": {"T1": t1_pca_deb, "T2": t2_pca_deb, "A1": a1_pca_deb, "A2": a2_pca_deb},
                 "NotFound": not_found, "Deleted": deleted})
        else:
            lex_dict = dict_to_json(lex_dict)
            response = json.dumps(
                {"Space": space, "Model": method, "Lower": lower, "PCA": pca,
                 "UsedAugmentations": {"A1": aug1_list, "A2": aug2_list},
                 "BiasedSpace:": {"T1": t1, "T2": t2, "A1": a1, "A2": a2},
                 "DebiasedSpace": {"T1": t1_deb, "T2": t2_deb, "A1": a1_deb, "A2": a2_deb},
                 "BiasedSpacePCA": {"T1": t1_pca_bias, "T2": t2_pca_bias, "A1": a1_pca_bias, "A2": a2_pca_bias},
                 "DebiasedSpacePCA": {"T1": t1_pca_deb, "T2": t2_pca_deb, "A1": a1_pca_deb, "A2": a2_pca_deb},
                 "NotFound": not_found, "Deleted": deleted, "LexDictionary": lex_dict})

    return response


def json_with_vector_data(content):
    target1, target2, attributes1, attributes2, augments1, augments2 = [], [], [], [], [], []
    if 'DebiasedSpace' in content:
        target1 = content['DebiasedSpace']['T1']
        target2 = content['DebiasedSpace']['T2']
        attributes1 = content['DebiasedSpace']['A1']
        attributes2 = content['DebiasedSpace']['A2']
        if 'Augmentations1' in content['DebiasedSpace']:
            augments1 = content['DebiasedSpace']['Augmentations1']
        if 'augments1' in content['DebiasedSpace']:
            augments2 = content['DebiasedSpace']['Augmentations2']
    return target1, target2, attributes1, attributes2, augments1, augments2


def json_lex_vector_data(content):
    print('JSON_Controller -- JSON LEX VECTOR DATA')
    if 'LexDictionary' in content:
        return content['LexDictionary']
