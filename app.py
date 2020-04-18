import datetime
import logging
import os

import augmentation_retrieval
import vector_retrieval
import json_controller

from flask import Flask, request
from flask import jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from bias_evaluation import evaluation_controller

''' RestAPI '''
# FLASK, CORS & Logging configuration
UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = {'txt', 'vec', 'vocab', 'vector'}
MAX_CONTENT_LENGTH = 250 * 1024 * 1024

app = Flask(__name__)
CORS(app)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


# logging.basicConfig(filename="logfiles.log", level=logging.INFO)
# print("logging configured")


# API-Connection Test
@app.route('/REST/', methods=['GET'])
def test():
    return 'CONNECTION WORKS'


# Retrieval of word vector representations for single words
# Example: http://127.0.0.1:5000/REST/retrieve_single_vector?embedding_space=fasttext&word=car
@app.route('/REST/vectors/single', methods=['GET'])
def retrieve_single_vector():
    # logging.info("APP:" + str(datetime.datetime.now()) + " Retrieve single vector is called")
    bar = request.args.to_dict()
    response, status_code = vector_retrieval.retrieve_vector('single', None, bar)
    return response, status_code


# Retrieval of word vector representations for a list of words
@app.route('/REST/vectors/multiple', methods=['POST'])
def retrieve_multiple_vectors():
    # logging.info("APP: " + str(datetime.datetime.now()) + " Retrieve multiple vectors is called")
    content = request.get_json()
    bar = request.args.to_dict()
    response, status_code = vector_retrieval.retrieve_vector('multiple', content, bar)
    return response, status_code


# Retrieves four augmentations for a word
@app.route('/REST/augmentations/single', methods=['GET'])
def retrieve_single_augmentation():
    bar = request.args.to_dict()
    response, status_code = augmentation_retrieval.retrieve_augmentations('single', None, bar)
    return response, status_code


# Retrieves 4 augmentations for a list of words
@app.route('/REST/augmentations/multiple', methods=['POST'])
def retrieve_multiple_augmentations():
    content = request.get_json()
    bar = request.args.to_dict()
    response, status_code = augmentation_retrieval.retrieve_augmentations('multiple', content, bar)
    return response, status_code


# Evaluates a bias specification with all implemented evaluation methods
@app.route('/REST/bias-evaluation/all', methods=['POST'])
def bias_evaluations_all():
    content = request.get_json()
    bar = request.args.to_dict()
    response, status_code = evaluation_controller.evaluation('all', content, bar)

    return response, status_code


# Evaluates a bias specification with the Embedding Coherence Test (ECT)
@app.route('/REST/bias-evaluation/ect', methods=['POST'])
def bias_evaluations_ect():
    content = request.get_json()
    bar = request.args.to_dict()
    response, status_code = evaluation_controller.evaluation('ect', content, bar)

    return response, status_code


# Evaluates a bias specification with the Bias Analogy Test (BAT)
@app.route('/REST/bias-evaluation/bat', methods=['POST'])
def bias_evaluations_bat():
    content = request.get_json()
    bar = request.args.to_dict()
    response, status_code = evaluation_controller.evaluation('bat', content, bar)

    return response, status_code


# Evaluates a bias specification with the Word Embedding Association Test (WEAT)
@app.route('/REST/bias-evaluation/weat', methods=['POST'])
def bias_evaluations_weat():
    content = request.get_json()
    bar = request.args.to_dict()
    response, status_code = evaluation_controller.evaluation('weat', content, bar)

    return response, status_code


# Evaluates a bias specification with K-Means++ clustering
@app.route('/REST/bias-evaluation/kmeans', methods=['POST'])
def bias_evaluations_kmeans():
    content = request.get_json()
    bar = request.args.to_dict()
    response, status_code = evaluation_controller.evaluation('kmeans', content, bar)

    return response, status_code


# General Bias-Direction Debiasing of a bias specifiication returning values
@app.route('/REST/debiasing/gbdd', methods=['POST'])
def debiasing_gbdd():
    return 200


# Bias Analogy Model debiasing of a bias specifiication returning values
@app.route('/REST/debiasing/bam', methods=['POST'])
def debiasing_bam():
    return 200


# Debiasing of bias specifications using GBDD and BAM, returning values
@app.route('/REST/debiasing/full/gbddxbam', methods=['POST'])
def debiasing_gbdd_bam():
    return 200


# Debiasing of bias specifications using BAM and GBDD, returning values in full size
@app.route('/REST/debiasing/full/bamxgbdd', methods=['POST'])
def debiasing_bam_gbdd():
    return 200


# Upload of complete embedding spaces
@app.route('/REST/uploads/embedding-spaces', methods=['POST'])
def upload_embedding_space():
    logging.info("APP: Receiving file from upload " + str(datetime.datetime.now()))
    print('Receiving file form upload')

    if 'uploadFile' not in request.files:
        resp = jsonify({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp
    file = request.files['uploadFile']
    # print(file.read())
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
        resp = jsonify({'message': 'Allowed file types are txt, vec or vocab'})
        resp.status_code = 400
        print('Case 4')
        return resp


# Check if uploaded file-name is accepted
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run()
