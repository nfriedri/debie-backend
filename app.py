import datetime
import logging
import os

import augmentation_retrieval
import data_controller
import upload_controller
import vector_retrieval

from flask import Flask, request
from flask import jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from bias_evaluation import evaluation_controller
from debiasing import debiasing_controller

''' RestAPI '''
# FLASK, CORS & Logging configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'vec', 'vocab', 'vectors'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024

app = Flask(__name__)
CORS(app)
#cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


# logging.basicConfig(filename="logfile.log", level=logging.INFO)
logging.info("APP: APP started at " + str(datetime.datetime.now()))
print("logging configured")


# API-Connection Test
@app.route('/REST/', methods=['GET'])
def test():
    return 'CONNECTION WORKS'


# Retrieval of word vector representations for single words
# Example: http://127.0.0.1:5000/REST/retrieve_single_vector?embedding_space=fasttext&word=car
@app.route('/REST/vectors/single', methods=['GET'])
def retrieve_single_vector():
    logging.info("APP:" + str(datetime.datetime.now()) + " Retrieve single vector is called")
    bar = request.args.to_dict()
    response, status_code = vector_retrieval.retrieve_vector('single', None, bar)
    return response, status_code


# Retrieval of word vector representations for a list of words
@app.route('/REST/vectors/multiple', methods=['POST'])
def retrieve_multiple_vectors():
    logging.info("APP: " + str(datetime.datetime.now()) + " Retrieve multiple vectors is called")
    content = request.get_json()
    bar = request.args.to_dict()
    response, status_code = vector_retrieval.retrieve_vector('multiple', content, bar)
    return response, status_code


# Retrieves four augmentations for a word
@app.route('/REST/augmentations/single', methods=['GET'])
def retrieve_single_augmentation():
    logging.info("APP: " + str(datetime.datetime.now()) + " Retrieve single augmentation is called")
    bar = request.args.to_dict()
    response, status_code = augmentation_retrieval.retrieve_augmentations('single', None, bar)
    return response, status_code


# Retrieves 4 augmentations for a list of words
@app.route('/REST/augmentations/multiple', methods=['POST'])
def retrieve_multiple_augmentations():
    logging.info("APP: " + str(datetime.datetime.now()) + " Retrieve multiple augmentations is called")
    content = request.get_json()
    bar = request.args.to_dict()
    response, status_code = augmentation_retrieval.retrieve_augmentations('multiple', content, bar)
    return response, status_code


# Evaluates a bias specification with all implemented evaluation methods
@app.route('/REST/bias-evaluation/all', methods=['POST'])
def bias_evaluations_all():
    logging.info("APP: " + str(datetime.datetime.now()) + " Bias Evaluation with ALL scores started")
    # print("APP: " + str(datetime.datetime.now()) + " Bias Evaluation with ALL scores started")
    content = request.get_json()
    bar = request.args.to_dict()
    response, status_code = evaluation_controller.evaluation('all', content, bar)

    return response, status_code


# Evaluates a bias specification with the Embedding Coherence Test (ECT)
@app.route('/REST/bias-evaluation/ect', methods=['POST'])
def bias_evaluations_ect():
    logging.info("APP: " + str(datetime.datetime.now()) + " Bias Evaluation with ECT scores started")
    content = request.get_json()
    bar = request.args.to_dict()
    response, status_code = evaluation_controller.evaluation('ect', content, bar)

    return response, status_code


# Evaluates a bias specification with the Bias Analogy Test (BAT)
@app.route('/REST/bias-evaluation/bat', methods=['POST'])
def bias_evaluations_bat():
    logging.info("APP: " + str(datetime.datetime.now()) + " Bias Evaluation with BAT scores started")
    content = request.get_json()
    bar = request.args.to_dict()
    response, status_code = evaluation_controller.evaluation('bat', content, bar)

    return response, status_code


# Evaluates a bias specification with the Word Embedding Association Test (WEAT)
@app.route('/REST/bias-evaluation/weat', methods=['POST'])
def bias_evaluations_weat():
    logging.info("APP: " + str(datetime.datetime.now()) + " Bias Evaluation with WEAT scores started")
    content = request.get_json()
    bar = request.args.to_dict()
    response, status_code = evaluation_controller.evaluation('weat', content, bar)

    return response, status_code


# Evaluates a bias specification with K-Means++ clustering
@app.route('/REST/bias-evaluation/kmeans', methods=['POST'])
def bias_evaluations_kmeans():
    logging.info("APP: " + str(datetime.datetime.now()) + " Bias Evaluation with KMEANS scores started")
    content = request.get_json()
    bar = request.args.to_dict()
    response, status_code = evaluation_controller.evaluation('kmeans', content, bar)

    return response, status_code


# Evaluates a bias specification with SVM-Classifier
@app.route('/REST/bias-evaluation/svm', methods=['POST'])
def bias_evaluations_svm():
    logging.info("APP: " + str(datetime.datetime.now()) + " Bias Evaluation with SVM-Classifier scores started")
    content = request.get_json()
    bar = request.args.to_dict()
    response, status_code = evaluation_controller.evaluation('svm', content, bar)

    return response, status_code


#
@app.route('/REST/bias-evaluation/simlex', methods=['POST'])
def bias_evaluations_simlex():
    logging.info("APP: " + str(datetime.datetime.now()) + " Semantic Quality Test SimLex started")
    content = request.get_json()
    bar = request.args.to_dict()
    response, status_code = evaluation_controller.evaluation('simlex', content, bar)

    return response, status_code


#
@app.route('/REST/bias-evaluation/wordsim', methods=['POST'])
def bias_evaluations_wordsim():
    logging.info("APP: " + str(datetime.datetime.now()) + " Semantic Quality Test WordSim started")
    content = request.get_json()
    bar = request.args.to_dict()
    response, status_code = evaluation_controller.evaluation('wordsim', content, bar)

    return response, status_code


# General Bias-Direction Debiasing of a bias specifiication returning values
@app.route('/REST/debiasing/gbdd', methods=['POST'])
def debiasing_gbdd():
    logging.info("APP: " + str(datetime.datetime.now()) + " GBDD Debiasing started")
    # print("APP: " + str(datetime.datetime.now()) + " GBDD Debiasing started")
    content = request.get_json()
    bar = request.args.to_dict()
    response, status_code = debiasing_controller.debiasing('gbdd', content, bar)

    return response, status_code


# Bias Analogy Model debiasing of a bias specifiication returning values
@app.route('/REST/debiasing/bam', methods=['POST'])
def debiasing_bam():
    logging.info("APP: " + str(datetime.datetime.now()) + " BAM Debiasing started")
    content = request.get_json()
    bar = request.args.to_dict()
    response, status_code = debiasing_controller.debiasing('bam', content, bar)

    return response, status_code


# Debiasing of bias specifications using GBDD and BAM, returning values
@app.route('/REST/debiasing/full/gbddxbam', methods=['POST'])
def debiasing_gbdd_bam():
    logging.info("APP: " + str(datetime.datetime.now()) + " GBDDxBAM Debiasing started")
    content = request.get_json()
    bar = request.args.to_dict()
    response, status_code = debiasing_controller.debiasing('gbddXbam', content, bar)

    return response, status_code


# Debiasing of bias specifications using BAM and GBDD, returning values in full size
@app.route('/REST/debiasing/full/bamxgbdd', methods=['POST'])
def debiasing_bam_gbdd():
    logging.info("APP: " + str(datetime.datetime.now()) + " BAMxGBDD Debiasing started")
    content = request.get_json()
    bar = request.args.to_dict()
    response, status_code = debiasing_controller.debiasing('bamXgbdd', content, bar)

    return response, status_code


# Upload of complete embedding spaces
@app.route('/REST/uploads/embedding-spaces', methods=['POST'])
def upload_embedding_space():
    logging.info("APP: Receiving file from upload " + str(datetime.datetime.now()))
    # print('Receiving file from upload')

    if 'vectorFile' in request.files:
        file = request.files['vectorFile']
        if file.filename == '':
            resp = jsonify({'message': 'No file selected for uploading'})
            resp.status_code = 402
            upload_controller.uploaded_binary = ''
            return resp
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            resp = jsonify({'message': 'File successfully uploaded'})
            upload_controller.uploaded_binary = 'false'
            resp.status_code = 201
            return resp
        else:
            resp = jsonify({'message': 'Allowed file types are txt, vec, vector or vocab'})
            resp.status_code = 400
            upload_controller.uploaded_binary = ''
            return resp

    if 'vocab' and 'vecs' in request.files:
        vocab = request.files['vocab']
        vecs = request.files['vecs']
        if vocab.filename and vecs.filename == '':
            resp = jsonify({'message': 'No files selected for uploading'})
            resp.status_code = 402
            upload_controller.uploaded_binary = ''
            return resp
        if vocab.filename and allowed_file(vocab.filename) and vecs.filename and allowed_file(vecs.filename):
            vocab_filename = secure_filename(vocab.filename)
            vocab.save(os.path.join(app.config['UPLOAD_FOLDER'], vocab_filename))
            vecs_filename = secure_filename(vecs.filename)
            vecs.save(os.path.join(app.config['UPLOAD_FOLDER'], vecs_filename))
            resp = jsonify({'message': 'Files successfully uploaded'})
            upload_controller.uploaded_binary = 'true'
            resp.status_code = 201
            return resp
        else:
            resp = jsonify({'message': 'Allowed file types are txt, vec, vector or vocab'})
            resp.status_code = 401
            upload_controller.uploaded_binary = ''
            return resp

    if 'vectorFile' and ('vocab' and 'vecs') not in request.files:
        resp = jsonify({'message': 'No file(s) part of the request'})
        resp.status_code = 400
        return resp


@app.route('/REST/uploads/initialize', methods=['GET'])
def initialize_uploaded_embeddings():
    logging.info("APP: " + str(datetime.datetime.now()) + " Initializing uploaded file(s)")
    bar = request.args.to_dict()
    # print(upload_controller.uploaded_binary)
    if upload_controller.uploaded_binary == 'true':
        vocab = bar['vocab']
        vecs = bar['vecs']
        data_controller.load_binary_uploads(vocab, vecs)
        resp = jsonify({'message': 'INITIALIZED BINARY VOCAB AND VEC FILE SUCCESSFULLY'})
        resp.status_code = 200
        return resp, resp.status_code
    if upload_controller.uploaded_binary == 'false':
        file = bar['file']
        data_controller.load_dict_uploaded_file(file)
        resp = jsonify({'message': 'INITIALIZED VECTOR FILE SUCCESSFULLY'})
        resp.status_code = 200
        return resp, resp.status_code
    else:
        resp = jsonify({'message': 'NO UPLOADED FILE(S) FOUND'})
        resp.status_code = 404
    return resp, resp.status_code


@app.route('/REST/uploads/delete', methods=['DELETE'])
def delete_uploaded_file():
    bar = request.args.to_dict()
    filename = bar['file']
    path = 'uploads/' + filename
    logging.info("APP: " + str(datetime.datetime.now()) + " Deleting uploaded file")
    try:
        os.remove(path)
        resp = jsonify({'message': 'REMOVED FILE SUCCESFULLY'})
    except FileNotFoundError:
        resp = jsonify({'message': 'FILE NOT FOUND'})
        return resp, 404
    return resp, 200


# Check if uploaded file-name is accepted
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run()
