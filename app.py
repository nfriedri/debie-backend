import json
import logging
import os
import sys

from flask import Flask, request, flash, redirect, url_for
from flask import jsonify
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

import JSONFormatter
import bias_eval_methods
import calculation
import database_handler
from debiasing import gbdd, bam, bam2

'''Initialize vectors into test and argument sets'''
'''

t1= ["glovers", "gladiolus", "nance", "crowfoot"]
t2= ["caterpillars", "gnats", "termites"]
a1= ["donation", "liberty", "tranquility", "fortunate", "mild"]
a2= ["misuse", "collision", "stench", "destitution", "demise"]


word_list1 = ["car", "plane", "bmw", "mercedes", "audi"]
word_list2 = ["football", "basketball", "adidas", "nike", "puma"]
word_list3 = ["aster", "clover", "hyacinth", "marigold", "poppy", "azalea", "crocus", "iris", "orchid", "rose",
              "daffodil", "lilac", "pansy", "tulip", "buttercup", "daisy", "lily", "peony", "violet",
              "carnation", "Gladiola", "magnolia", "petunia"]

dict1 = database_handler.get_multiple_vectors_from_db(t1, 'fasttext')
dict2 = database_handler.get_multiple_vectors_from_db(t2, 'fasttext')
print(len(dict1))
print(len(dict2))
print('START')
print(bam2.bias_alignment_model2(dict1, dict2))

'''


''' RestAPI '''
# FLASK, CORS & Logging configuration

UPLOAD_FOLDER = 'C:\\Users\\Niklas Friedrich\\Documents\\GitHub\\debie_backend\\uploads\\files'
ALLOWED_EXTENSIONS = {'txt', 'vec'}

app = Flask(__name__)
CORS(app)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


logging.basicConfig(filename="logfiles.log", level=logging.INFO)
print("logging configured")


# Example: http://127.0.0.1:5000/REST/retrieve_single_vector?embedding_space=fasttext&word=car
@app.route('/REST/vectors/single', methods=['GET'])
def retrieve_single_vector():
    logging.info("APP: Retrieve single vector is called")
    bar = request.args.to_dict()
    space = bar['space']
    search = bar['word']
    vector_dict = database_handler.get_vector_from_database(search, space)
    response = jsonify(word=[word for word in vector_dict], vector=[list(vector_dict[vec]) for vec in vector_dict])
    logging.info("APP: Retrieved vector")
    return response


@app.route('/REST/vectors/multiple', methods=['POST'])
def retrieve_multiple_vectors():
    logging.info("APP: Retrieve multiple vectors is called")
    bar = request.args.to_dict()
    space = bar['space']
    content = request.get_json()
    word_list = content['data'].split(' ')
    vector_dict = database_handler.get_multiple_vectors_from_db(word_list, space)
    response = jsonify(word=[word for word in vector_dict], vector=[list(vector_dict[vec]) for vec in vector_dict])
    logging.info("APP: Retrieved vectors")
    return response


@app.route('/REST/augmentations/single')
def retrieve_single_augmentation():
    return 200, 'OK'


@app.route('/REST/augmentations/first10k')
def retrieve_multiple_augmentations_10k():
    return 200, 'OK'


@app.route('/REST/bias-evaluation', methods=['POST'])
def bias_evaluations():
    logging.info("APP: Bias Evaluation is called")
    content = request.get_json()
    methods = content['Method']
    logging.info("APP: Starting evaluation process")
    target1, target2, arg1, arg2 = JSONFormatter.retrieve_vectors_from_db(content)
    target1, target2 = calculation.check_sizes(target1, target2)
    arg1, arg2 = calculation.check_sizes(arg1, arg2)
    if len(target1) == 0 or len(target2) == 0 or len(arg1) == 0 or len(arg2) == 0:
        logging.info("APP: Stopped, no values found in database")
        return jsonify(message="ERROR: No values found in database."), 400
    logging.info("APP: Final retrieved set sizes: T1=" + str(len(target1)) + " T2=" + str(len(target2)) + " A1=" + str(
    len(arg1)) + " A2=" + str(len(arg2)))
    logging.info("APP: Evaluation process started")
    try:
        result = bias_eval_methods.return_bias_evaluation(methods, target1, target2, arg1, arg2)
    except:
        return "500 ERROR vector dimensions are unequal"
    return result


@app.route('/REST/bias-evaluation/all', methods=['POST'])
def bias_evaluations_all():
    logging.info("APP: Bias Evaluation ALL Methods is called")
    content = request.get_json()
    database = request.args.to_dict()['space']
    logging.info("APP: Starting evaluation process")
    target1, target2, arg1, arg2 = JSONFormatter.retrieve_vectors_from_db(content, database)
    target1, target2 = calculation.check_sizes(target1, target2)
    arg1, arg2 = calculation.check_sizes(arg1, arg2)
    if len(target1) == 0 or len(target2) == 0 or len(arg1) == 0 or len(arg2) == 0:
        logging.info("APP: Stopped, no values found in database")
        return jsonify(message="ERROR: No values found in database."), 400
    logging.info("APP: Final retrieved set sizes: T1=" + str(len(target1)) + " T2=" + str(len(target2)) + " A1=" + str(
        len(arg1)) + " A2=" + str(len(arg2)))
    logging.info("APP: Evaluation process started")
    try:
        result = bias_eval_methods.return_eval_all(target1, target2, arg1, arg2)
    except:
        return jsonify(message="Internal Server Error"), 500
    return result, 200


@app.route('/REST/bias-evaluation/ect', methods=['POST'])
def bias_evaluations_ect():
    logging.info("APP: Bias Evaluation ECT Method is called")
    content = request.get_json()
    logging.info("APP: Starting evaluation process")
    target1, target2, arg1, arg2 = JSONFormatter.retrieve_vectors_from_db(content)
    target1, target2 = calculation.check_sizes(target1, target2)
    arg1, arg2 = calculation.check_sizes(arg1, arg2)
    if len(target1) == 0 or len(target2) == 0 or len(arg1) == 0 or len(arg2) == 0:
        logging.info("APP: Stopped, no values found in database")
        return jsonify(message="ERROR: No values found in database."), 400
    logging.info("APP: Final retrieved set sizes: T1=" + str(len(target1)) + " T2=" + str(len(target2)) + " A1=" + str(
        len(arg1)) + " A2=" + str(len(arg2)))
    logging.info("APP: Evaluation process started")
    try:
        result = bias_eval_methods.return_eval_ect(target1, target2, arg1, arg2)
    except:
        return jsonify(message="Internal Server Error"), 500
    return result, 200


@app.route('/REST/bias-evaluation/bat', methods=['POST'])
def bias_evaluations_bat():
    logging.info("APP: Bias Evaluation BAT Method is called")
    content = request.get_json()
    logging.info("APP: Starting evaluation process")
    target1, target2, arg1, arg2 = JSONFormatter.retrieve_vectors_from_db(content)
    target1, target2 = calculation.check_sizes(target1, target2)
    arg1, arg2 = calculation.check_sizes(arg1, arg2)
    if len(target1) == 0 or len(target2) == 0 or len(arg1) == 0 or len(arg2) == 0:
        logging.info("APP: Stopped, no values found in database")
        return jsonify(message="ERROR: No values found in database."), 400
    logging.info("APP: Final retrieved set sizes: T1=" + str(len(target1)) + " T2=" + str(len(target2)) + " A1=" + str(
        len(arg1)) + " A2=" + str(len(arg2)))
    logging.info("APP: Evaluation process started")
    try:
        result = bias_eval_methods.return_eval_bat(target1, target2, arg1, arg2)
    except:
        return jsonify(message="Internal Server Error"), 500
    return result, 200


@app.route('/REST/bias-evaluation/weat', methods=['POST'])
def bias_evaluations_weat():
    logging.info("APP: Bias Evaluation WEAT Method is called")
    content = request.get_json()
    logging.info("APP: Starting evaluation process")
    target1, target2, arg1, arg2 = JSONFormatter.retrieve_vectors_from_db(content)
    target1, target2 = calculation.check_sizes(target1, target2)
    arg1, arg2 = calculation.check_sizes(arg1, arg2)
    if len(target1) == 0 or len(target2) == 0 or len(arg1) == 0 or len(arg2) == 0:
        logging.info("APP: Stopped, no values found in database")
        return jsonify(message="ERROR: No values found in database."), 400
    logging.info("APP: Final retrieved set sizes: T1=" + str(len(target1)) + " T2=" + str(len(target2)) + " A1=" + str(
        len(arg1)) + " A2=" + str(len(arg2)))
    logging.info("APP: Evaluation process started")
    try:
        result = bias_eval_methods.return_eval_weat(target1, target2, arg1, arg2)
    except:
        return jsonify(message="Internal Server Error"), 500
    return result, 200


@app.route('/REST/bias-evaluation/kmeans', methods=['POST'])
def bias_evaluations_kmeans():
    logging.info("APP: Bias Evaluation KMEANS Method is called")
    content = request.get_json()
    logging.info("APP: Starting evaluation process")
    target1, target2, arg1, arg2 = JSONFormatter.retrieve_vectors_from_db(content)
    target1, target2 = calculation.check_sizes(target1, target2)
    if len(target1) == 0 or len(target2) == 0:
        logging.info("APP: Stopped, no values found in database")
        return jsonify(message="ERROR: No values found in database."), 400
    logging.info("APP: Final retrieved set sizes: T1=" + str(len(target1)) + " T2=" + str(len(target2)))
    logging.info("APP: Evaluation process started")
    try:
        result = bias_eval_methods.return_eval_kmeans(target1, target2)
    except:
        return jsonify(message="Internal Server Error"), 500
    return result, 200


@app.route('/REST/debiasing/full/gbdd', methods=['POST'])
def debiasing_full_gbdd():
    logging.info("APP: Debiasing is called")
    # Get content from JSON
    content = request.get_json()
    embedding_space = content['EmbeddingSpace']
    methods = content['Method']
    logging.info("APP: Starting evaluation in " + str(embedding_space) + "embedding space with " + str(methods))

    # Retrieve & check Vectors from database
    target1, target2, arg1, arg2 = JSONFormatter.retrieve_vectors_from_db(content)
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


@app.route('/REST/debiasing/pca/gbdd', methods=['POST'])
def debiasing_pca_gbdd():
    logging.info("APP: Debiasing is called")
    content = request.get_json()
    embedding_space = content['EmbeddingSpace']
    methods = content['Method']
    logging.info("APP: Starting debiasing in " + str(embedding_space) + " embedding space with " + str(methods))

    # Retrieve & check vectors from database
    target1, target2, arg1, arg2 = JSONFormatter.retrieve_vectors_from_db(content)
    target1, target2 = calculation.check_sizes(target1, target2)
    arg1, arg2 = calculation.check_sizes(arg1, arg2)
    logging.info("APP: Retrieved Vectors from database")
    logging.info("APP: Final retrieved set sizes: T1=" + str(len(target1)) + " T2=" + str(len(target2)) + " A1=" + str(
        len(arg1)) + " A2=" + str(len(arg2)))

    logging.info("APP: Debiasing process started")
    debiased1, debiased2 = gbdd.generalized_bias_direction_debiasing(target1, target2)
    debiased1_copy, debiased2_copy = calculation.create_duplicates(debiased1, debiased2)
    debiased = debiased1_copy.update(debiased2_copy)
    target1_copy, target2_copy = calculation.create_duplicates(target1, target2)
    target = target1_copy.update(target2_copy)

    biased_pca = calculation.principal_composant_analysis(target1, target2)
    debiased_pca = calculation.principal_composant_analysis(debiased1, debiased2)

    response = json.dumps(
        {"EmbeddingSpace": embedding_space, "Method": methods,
         "BiasedVectorsPCA": JSONFormatter.dict_to_json(biased_pca),
         "DebiasedVectorsPCA": JSONFormatter.dict_to_json(debiased_pca),
         "BiasedVecs:": JSONFormatter.dict_to_json(debiased),
         "DebiasedVecs": JSONFormatter.dict_to_json(target)})
    logging.info("APP: Debiasing process with PCA finished")
    logging.info("APP: " + str(response))
    # print(response)
    return response


@app.route('/REST/debiasing/full/bam', methods=['POST'])
def debiasing_full_bam():
    return 200


@app.route('/REST/debiasing/pca/bam', methods=['POST'])
def debiasing_pca_bam():
    return 200


@app.route('/REST/uploads/embedding-spaces', methods=['PUT'])
def upload_embedding_space():
    logging.info("APP: Receiving file form upload")
    print('Receiving file form upload')

    if 'uploadFile' not in request.files:
        resp = jsonify({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp
    file = request.files['uploadFile']
    print(file.read())
    if file.filename == '':
        resp = jsonify({'message': 'No file selected for uploading'})
        resp.status_code = 400
        return resp
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        resp = jsonify({'message': 'File successfully uploaded'})
        resp.status_code = 201
        print('Case 3')
        return resp
    else:
        resp = jsonify({'message': 'Allowed file types are txt, pdf, png, jpg, jpeg, gif'})
        resp.status_code = 400
        print('Case 4')
        return resp


@app.route('/REST/uploads/json/format1', methods=['PUT'])
def upload_json_format1():
    return 200


@app.route('/REST/uploads/json/format2', methods=['PUT'])
def upload_json_format2():
    return 200


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run()
