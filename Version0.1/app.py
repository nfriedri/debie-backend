import datetime
import logging
import os

from flask import Flask, request
from flask import jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

import JSONFormatter
import database_handler
from bias_evaluation import bias_eval_methods
from debiasing import debiasing_models

''' RestAPI '''
# FLASK, CORS & Logging configuration

# UPLOAD_FOLDER = 'C:\\Users\\Niklas Friedrich\\Documents\\GitHub\\debie_backend\\uploads\\files'  # Change required
UPLOAD_FOLDER = '/home/nfriedri/debie-backend/uploads/files' # Remove for server deployment
ALLOWED_EXTENSIONS = {'txt', 'vec'}
MAX_CONTENT_LENGTH = 250 * 1024 * 1024

app = Flask(__name__)
CORS(app)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

logging.basicConfig(filename="logfiles.log", level=logging.INFO)
print("logging configured")


# API-Connection Test
@app.route('/REST/', methods=['GET'])
def test():
    return 'CONNECTION WORKS'


# Check if JSON-upload works
@app.route('/REST/uploads/validJSON', methods=['POST'])
def valid_JSON():
    print('Valid JSON called')
    content = request.get_json()
    t1, t2, a1, a2 = JSONFormatter.retrieve_vectors_from_json_evaluation(content)
    print(t1)
    print(t2)
    print(a1)
    print(a2)
    return 'VALID', 200


# Retrieval of word vector representations for single words
# Example: http://127.0.0.1:5000/REST/retrieve_single_vector?embedding_space=fasttext&word=car
@app.route('/REST/vectors/single', methods=['GET'])
def retrieve_single_vector():
    logging.info("APP:"  + str(datetime.datetime.now()) + " Retrieve single vector is called")
    bar = request.args.to_dict()
    space = bar['space']
    search = bar['word']
    try:
        vector_dict = database_handler.get_vector_from_database(search, space)
        response = jsonify(word=[word for word in vector_dict], vector=[list(vector_dict[vec]) for vec in vector_dict])
        logging.info("APP: Retrieved vector")
    except:
        return jsonify(message="NOT FOUND"), 404
    return response, 200


# Retrieval of word vector representations for a list of words
@app.route('/REST/vectors/multiple', methods=['POST'])
def retrieve_multiple_vectors():
    logging.info("APP: " + str(datetime.datetime.now()) + " Retrieve multiple vectors is called")
    bar = request.args.to_dict()
    space = bar['space']
    content = request.get_json()
    word_list = content['data'].split(' ')
    try:
        vector_dict = database_handler.get_multiple_vectors_from_db(word_list, space)
        response = jsonify(word=[word for word in vector_dict], vector=[list(vector_dict[vec]) for vec in vector_dict])
        logging.info("APP: Retrieved vectors")
    except:
        return jsonify(message="NOT FOUND"), 404
    return response, 200


# Retrieves four augmentations for a word
@app.route('/REST/augmentations/single', methods=['GET'])
def retrieve_single_augmentation():
    logging.info("APP: " + str(datetime.datetime.now()) + " Retrieve single augmentation is called")
    bar = request.args.to_dict()
    search = bar['word']
    try:
        augmentations = database_handler.get_augmentation_from_db(search)
        response = jsonify(word=search, augments=[augmentations[i] for i in range(len(augmentations))])
        logging.info("APP: Retrieved vector")
    except:
        return jsonify(message="NOT FOUND"), 404
    return response, 200


# Retrieves 4 augmentations for a list of words
@app.route('/REST/augmentations/multiple', methods=['POST'])
def retrieve_multiple_augmentations():
    logging.info("APP: " + str(datetime.datetime.now()) + " Retrieve multiple augmentations is called")
    bar = request.args.to_dict()
    content = request.get_json()
    word_list = content['data'].split(' ')
    try:
        augmentations = database_handler.get_multiple_augmentation_from_db(word_list)
        print(word for word in augmentations)
        print(augmentations[word] for word in augmentations)
        response = jsonify(words=[word for word in word_list],
                           augments=[list(augmentations[word]) for word in augmentations])
        # response = json.dumps(augmentations)
        logging.info("APP: Retrieved vector")
    except:
        return jsonify(message="NOT FOUND"), 404
    return response, 200


# Evaluates a bias specification with all implemented evaluation methods
@app.route('/REST/bias-evaluation/all', methods=['POST'])
def bias_evaluations_all():
    logging.info("APP: " + str(datetime.datetime.now()) + " Bias Evaluation ALL Methods is called")
    content = request.get_json()
    records = request.args.to_dict()
    try:
        result = bias_eval_methods.return_bias_evaluation('all', records, content)
    except Exception as e:
        print(e)
        pass
        return jsonify(message="Internal Server Error"), 500
    return result, 200


# Evaluates a bias specification with the Embedding Coherence Test (ECT)
@app.route('/REST/bias-evaluation/ect', methods=['POST'])
def bias_evaluations_ect():
    logging.info("APP: " + str(datetime.datetime.now()) + " Bias Evaluation ECT Method is called")
    content = request.get_json()
    records = request.args.to_dict()
    try:
        result = bias_eval_methods.return_bias_evaluation('ect', records, content)
    except Exception as e:
        print(e)
        pass
        return jsonify(message="Internal Server Error"), 500
    return result, 200


# Evaluates a bias specification with the Bias Analogy Test (BAT)
@app.route('/REST/bias-evaluation/bat', methods=['POST'])
def bias_evaluations_bat():
    logging.info("APP: " + str(datetime.datetime.now()) + " Bias Evaluation BAT Method is called")
    content = request.get_json()
    records = request.args.to_dict()
    try:
        result = bias_eval_methods.return_bias_evaluation('bat', records, content)
    except Exception as e:
        print(e)
        pass
        return jsonify(message="Internal Server Error"), 500
    return result, 200


# Evaluates a bias specification with the Word Embedding Association Test (WEAT)
@app.route('/REST/bias-evaluation/weat', methods=['POST'])
def bias_evaluations_weat():
    logging.info("APP: " + str(datetime.datetime.now()) + " Bias Evaluation WEAT Method is called")
    content = request.get_json()
    records = request.args.to_dict()
    try:
        result = bias_eval_methods.return_bias_evaluation('weat', records, content)
    except Exception as e:
        print(e)
        pass
        return jsonify(message="Internal Server Error"), 500
    return result, 200


# Evaluates a bias specification with K-Means++ clustering
@app.route('/REST/bias-evaluation/kmeans', methods=['POST'])
def bias_evaluations_kmeans():
    logging.info("APP: " + str(datetime.datetime.now()) + " Bias Evaluation kMeans Method is called")
    content = request.get_json()
    records = request.args.to_dict()
    try:
        result = bias_eval_methods.return_bias_evaluation('kmeans', records, content)
    except Exception as e:
        print(e)
        pass
        return jsonify(message="Internal Server Error"), 500
    return result, 200


# General Bias-Direction Debiasing of a bias specifiication returning values in full size
@app.route('/REST/debiasing/full/gbdd', methods=['POST'])
def debiasing_full_gbdd():
    logging.info("APP: " + str(datetime.datetime.now()) + " GBDD Debiasing full is called")
    content = request.get_json()
    arguments = request.args.to_dict()
    methods = 'gbdd'
    response = debiasing_models.return_full_debiasing(methods, arguments, content)
    return response


# General Bias-Direction Debiasing of a bias specifiication returning compressed values
@app.route('/REST/debiasing/pca/gbdd', methods=['POST'])
def debiasing_pca_gbdd():
    logging.info("APP: " + str(datetime.datetime.now()) + " GBDD Debiasing PCA is called")
    content = request.get_json()
    arguments = request.args.to_dict()
    methods = 'gbdd'
    response = debiasing_models.return_pca_debiasing(methods, arguments, content)
    return response


# Bias Analogy Model debiasing of a bias specifiication returning values in full size
@app.route('/REST/debiasing/full/bam', methods=['POST'])
def debiasing_full_bam():
    logging.info("APP: " + str(datetime.datetime.now()) + " BAM Debiasing full is called")
    content = request.get_json()
    arguments = request.args.to_dict()
    methods = 'bam'
    response = debiasing_models.return_full_debiasing(methods, arguments, content)
    return response


# Bias Analogy Model debiasing of a bias specifiication returning compressed values
@app.route('/REST/debiasing/pca/bam', methods=['POST'])
def debiasing_pca_bam():
    logging.info("APP: " + str(datetime.datetime.now()) + " BAM Debiasing PCA is called")
    content = request.get_json()
    arguments = request.args.to_dict()
    methods = 'bam'
    response = debiasing_models.return_pca_debiasing(methods, arguments, content)
    return response


# Debiasing of bias specifications using GBDD and BAM, returning values in full size
@app.route('/REST/debiasing/full/gbddxbam', methods=['POST'])
def debiasing_full_gbdd_bam():
    logging.info("APP: " + str(datetime.datetime.now()) + " GBDD x BAM Debiasing full is called")
    # Get content from JSON
    content = request.get_json()
    arguments = request.args.to_dict()
    methods = 'gbddxbam'
    response = debiasing_models.return_full_debiasing(methods, arguments, content)
    return response


# Debiasing of bias specifications using GBDD and BAM, returning compressed values
@app.route('/REST/debiasing/pca/gbddxbam', methods=['POST'])
def debiasing_pca_gbdd_bam():
    logging.info("APP: " + str(datetime.datetime.now()) + " GBDD x BAM Debiasing PCA is called")
    content = request.get_json()
    arguments = request.args.to_dict()
    methods = 'gbddxbam'
    response = debiasing_models.return_pca_debiasing(methods, arguments, content)
    return response


# Debiasing of bias specifications using BAM and GBDD, returning values in full size
@app.route('/REST/debiasing/full/bamxgbdd', methods=['POST'])
def debiasing_full_bam_gbdd():
    logging.info("APP: " + str(datetime.datetime.now()) + " BAM x GBDD Debiasing full is called")
    content = request.get_json()
    arguments = request.args.to_dict()
    methods = 'bamxgbdd'
    response = debiasing_models.return_full_debiasing(methods, arguments, content)
    return response


# Debiasing of bias specifications using BAM and GBDD, returning compressed values
@app.route('/REST/debiasing/pca/bamxgbdd', methods=['POST'])
def debiasing_pca_bam_gbdd():
    logging.info("APP: " + str(datetime.datetime.now()) + " BAM x GBDD Debiasing PCA is called")
    content = request.get_json()
    arguments = request.args.to_dict()
    methods = 'bamxgbdd'
    response = debiasing_models.return_pca_debiasing(methods, arguments, content)
    return response


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


