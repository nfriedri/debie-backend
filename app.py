import json
import logging
from flask import Flask, request
from flask import jsonify
from flask_cors import CORS
import JSONFormatter
import bias_eval_methods
import calculation
from database_handler import get_multiple_vectors_from_db
from debiasing import gbdd, bam

'''Initialize vectors into test and argument sets'''
fasttext = "C:\\Users\\Niklas Friedrich\\Documents\\wiki-news-300d-1M.vec"

'''
test_files = ["C:\\Users\\Niklas Friedrich\\Documents\\Initial_Test\\initial_t1.vec",
              "C:\\Users\\Niklas Friedrich\\Documents\\Initial_Test\\initial_t2.vec",
              "C:\\Users\\Niklas Friedrich\\Documents\\Initial_Test\\initial_a1.vec",
              "C:\\Users\\Niklas Friedrich\\Documents\\Initial_Test\\initial_a2.vec"
              ]

init_t1 = vectors.load_vectors(test_files[0])
init_t2 = vectors.load_vectors(test_files[1])
init_a1 = vectors.load_vectors(test_files[2])
init_a2 = vectors.load_vectors(test_files[3])

t1= ["glovers", "gladiolus", "nance", "crowfoot"]
t2= ["caterpillars", "gnats", "termites", "avenger", "ants", "bumblebee"]
a1= ["donation", "liberty", "tranquility", "fortunate", "mild"]
a2= ["misuse", "collision", "stench", "destitution", "demise"]


word_list1 = ["car", "plane", "bmw", "mercedes", "audi"]
word_list2 = ["football", "basketball", "adidas", "nike", "puma"]
word_list3 = ["aster", "clover", "hyacinth", "marigold", "poppy", "azalea", "crocus", "iris", "orchid", "rose",
              "daffodil", "lilac", "pansy", "tulip", "buttercup", "daisy", "lily", "peony", "violet",
              "carnation", "Gladiola", "magnolia", "petunia"]
'''
''' RestAPI '''

app = Flask(__name__)
CORS(app)

logging.basicConfig(filename="logfiles.log", level=logging.INFO)
print("logging configured")


@app.route('/REST/POST_word', methods=["POST"])
def get_vector():
    content = request.get_json()
    data = content['data']
    vector = JSONFormatter.get_vector_from_json(data)
    vector_list = list(vector[data])
    return jsonify(word=data, vector=vector_list)


@app.route('/REST/POST_words', methods=["POST"])
def get_vectors():
    # Receive words from input field:
    content = request.get_json()
    data = content['data']
    # Split words from input field
    list_words = data.split(' ')
    vector_dict = JSONFormatter.get_multiple_vectors_from_json(list_words, 'fasttextdb')
    response = jsonify(word=[word for word in vector_dict], vector=[list(vector_dict[vec]) for vec in vector_dict])
    return response


@app.route('/REST/bias_evaluation', methods=['POST'])
def bias_evaluations():
    logging.info("APP: Bias Evaluation is called")
    # Get content from JSON
    content = request.get_json()
    embedding_space = content['EmbeddingSpace']
    methods = content['Method']
    logging.info("APP: Starting evaluation in " + str(embedding_space) + "embedding space with " + str(methods))
    # Retrieve & check vectors from database
    target1, target2, arg1, arg2 = JSONFormatter.retrieve_vectors(content, embedding_space)
    target1, target2 = calculation.check_sizes(target1, target2)
    arg1, arg2 = calculation.check_sizes(arg1, arg2)
    logging.info("APP: Retrieved Vectors from database")
    logging.info("APP: Final retrieved set sizes: T1=" + str(len(target1)) + " T2=" + str(len(target2)) + " A1=" + str(
        len(arg1)) + " A2=" + str(len(arg2)))

    logging.info("APP: Evaluation process started")
    return bias_eval_methods.return_bias_evaluation(methods, target1, target2, arg1, arg2)


@app.route('/REST/debiasing', methods=['POST'])
def debiasing():
    logging.info("APP: Debiasing is called")
    # Get content from JSON
    content = request.get_json()
    embedding_space = content['EmbeddingSpace']
    methods = content['Method']
    logging.info("APP: Starting evaluation in " + str(embedding_space) + "embedding space with " + str(methods))

    # Retrieve & check Vectors from database
    target1, target2, arg1, arg2 = JSONFormatter.retrieve_vectors(content, embedding_space)
    target1, target2 = calculation.check_sizes(target1, target2)
    arg1, arg2 = calculation.check_sizes(arg1, arg2)
    logging.info("APP: Retrieved Vectors from database")
    logging.info("APP: Final retrieved set sizes: T1=" + str(len(target1)) + " T2=" + str(len(target2)) + " A1=" + str(
        len(arg1)) + " A2=" + str(len(arg2)))

    logging.info("APP: Debiasing process started")
    # Following lines will be moved to debias_methods.py soon:
    result1, result2 = gbdd.generalized_bias_direction_debiasing(target1, target2)
    response = json.dumps(
        {"biased": JSONFormatter.dict_to_json(target1), "debiased": JSONFormatter.dict_to_json(result1)})
    # response = jsonify(GBDDVecs1=result1, GBDDVecs2=result2)
    logging.info("APP: Debiasing process finished")
    return response


@app.route('/REST/debiasing_visualization', methods=['POST'])
def debias_visualize():
    logging.info("APP: Debiasing is called")
    content = request.get_json()
    embedding_space = content['EmbeddingSpace']
    methods = content['Method']
    logging.info("APP: Starting evaluation in " + str(embedding_space) + "embedding space with " + str(methods))

    # Retrieve & check vectors from database
    target1, target2, arg1, arg2 = JSONFormatter.retrieve_vectors(content, embedding_space)
    target1, target2 = calculation.check_sizes(target1, target2)
    arg1, arg2 = calculation.check_sizes(arg1, arg2)
    logging.info("APP: Retrieved Vectors from database")
    logging.info("APP: Final retrieved set sizes: T1=" + str(len(target1)) + " T2=" + str(len(target2)) + " A1=" + str(
        len(arg1)) + " A2=" + str(len(arg2)))

    logging.info("APP: Debiasing process started")
    debiased1, debiased2 = gbdd.generalized_bias_direction_debiasing(target1, target2)

    biased_pca = calculation.principal_composant_analysis(target1, target2)
    debiased_pca = calculation.principal_composant_analysis(debiased1, debiased2)

    response = json.dumps(
        {"EmbeddingSpace": embedding_space, "Method": methods, "BiasedVectors": JSONFormatter.dict_to_json(biased_pca),
         "DebiasedVectors": JSONFormatter.dict_to_json(debiased_pca)})
    logging.info("APP: Debiasing process with PCA finished")
    logging.info("APP: " + str(response))
    print(response)
    return response


if __name__ == '__main__':
    app.run()


