import random

import JSONFormatter
import calculation
import database_handler
import vectors
from bias_evaluation import ect, bat, k_means, weat
from flask import Flask, request
from flask import jsonify
from flask_cors import CORS

from debiasing import gbdd, bam

'''Initialize vectors into test and argument sets'''
test_file = "C:\\Users\\Niklas Friedrich\\Documents\\wiki-news-300d-1M.vec"
test_files = ["C:\\Users\\Niklas Friedrich\\Documents\\Initial_Test\\initial_t1.vec",
              "C:\\Users\\Niklas Friedrich\\Documents\\Initial_Test\\initial_t2.vec",
              "C:\\Users\\Niklas Friedrich\\Documents\\Initial_Test\\initial_a1.vec",
              "C:\\Users\\Niklas Friedrich\\Documents\\Initial_Test\\initial_a2.vec"
              ]

# init_t1 = vectors.load_vectors(test_files[0])
# init_t2 = vectors.load_vectors(test_files[1])
# init_a1 = vectors.load_vectors(test_files[2])
# init_a2 = vectors.load_vectors(test_files[3])
'''
t1= ["glovers", "gladiolus", "nance", "crowfoot", "meadowsweet", "dianthus", "pinkish", "dolly", "poppies", "cyclamen"]
t2= ["caterpillars", "gnats", "termites", "avenger", "ants", "bumblebee", "arachnid", "sticking", "cricketing", "flit"]
a1= ["donation", "liberty", "tranquility", "fortunate", "mild", "laugh", "diamonds", "holiday", "truthful", "endowment"]
a2= ["misuse", "collision", "stench", "destitution", "demise", "anguish", "annihilate", "estrangement", "illness"]

t1= ["glovers", "gladiolus", "nance", "crowfoot"]
t2= ["caterpillars", "gnats", "termites", "avenger", "ants", "bumblebee"]
a1= ["donation", "liberty", "tranquility", "fortunate", "mild"]
a2= ["misuse", "collision", "stench", "destitution", "demise"]
'''
# print(weat.word_embedding_association_test(init_t1, init_t2, init_a1, init_a2, 100))


word_list1 = ["car", "plane", "BMW", "Mercedes", "Audi"]
word_list2 = ["football", "basketball", "adidas", "nike", "puma"]
word_list3 = ["aster", "clover", "hyacinth", "marigold", "poppy", "azalea", "crocus", "iris", "orchid", "rose",
              "daffodil", "lilac", "pansy", "tulip", "buttercup", "daisy", "lily", "peony", "violet",
              "carnation", "Gladiola", "magnolia", "petunia"]

# print("BAM")
# dict1 = database_handler.get_multiple_vectors_from_db(word_list1)
# dict2 = database_handler.get_multiple_vectors_from_db(word_list2)
# print(gbdd.generalized_bias_direction_debiasing(dict1, dict2))
# print(bam.bias_alignment_model(dict1, dict2))

# vectors2 = database_handler.get_multiple_vectors_from_db(word_list2)
# for word in vectors2:
#     print(list(vectors2[word]))


''' RestAPI '''

app = Flask(__name__)
CORS(app)


@app.route('/REST/POST_word', methods=["POST"])
def get_vector():
    content = request.get_json()
    data = content['data']
    print(data)
    vector = vectors.load_word(test_file, data)
    vector_list = list(vector[data])
    print(vector_list)
    return jsonify(word=data, vector=vector_list)


@app.route('/REST/POST_words', methods=["POST"])
def get_vectors():
    # Receive words from input field:
    content = request.get_json()
    data = content['data']
    # Split words from input field
    list_words = data.split(' ')
    vector_dict = vectors.load_multiple_words(test_file, list_words)
    response = jsonify(word=[word for word in vector_dict], vector=[list(vector_dict[vec]) for vec in vector_dict])
    print("JSON:")
    print(response)
    print("EOF")
#     word = [word for word in vector_dict]
    return response


@app.route('/REST/bias_evaluation', methods=['POST'])
def receive_bias_evaluations():
    # vector_space = request.args.get('vecspace')
    # eval_method = request.arg.get('method')
    # print(vector_space)
    # print(eval_method)
    content = request.get_json()

    test_vectors1, test_vectors2, arg_vectors1, arg_vectors2 = JSONFormatter.get_vector_from_json_via_db(content)
    test_vectors1, test_vectors2 = calculation.check_sizes(test_vectors1, test_vectors2)
    arg_vectors1, arg_vectors2 = calculation.check_sizes(arg_vectors1, arg_vectors2)
    print("Set Sizes:")
    print(len(test_vectors1))
    print(len(test_vectors2))
    print(len(arg_vectors1))
    print(len(arg_vectors2))
    ect_value1, p_value1 = ect.embedding_coherence_test(test_vectors1, test_vectors2, arg_vectors1)
    ect_value2, p_value2 = ect.embedding_coherence_test(test_vectors1, test_vectors2, arg_vectors2)
    #    bat_result = bat.biased_analogy_test(test_vectors1, test_vectors2, arg_vectors1, arg_vectors2)
    bat_result = 'Currently not available'
    weat_effect_size, weat_p_value = weat.word_embedding_association_test(test_vectors1, test_vectors2, arg_vectors1,
                                                                          arg_vectors2)

    response = jsonify(ect_value1=ect_value1, p_value1=p_value1, p_value2=p_value2, ect_value2=ect_value2,
                       bat_value=bat_result, weat_effect_size=weat_effect_size, weat_pvalue=weat_p_value)
    return response
