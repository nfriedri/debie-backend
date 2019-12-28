import io
import logging
import os
import traceback

from flask import Flask, request
from flask import jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

import JSONFormatter
import bias_eval_methods
import calculation
import database_handler
import debias_methods
import vectors


''' RestAPI '''
# FLASK, CORS & Logging configuration

UPLOAD_FOLDER = 'C:\\Users\\Niklas Friedrich\\Documents\\GitHub\\debie_backend\\uploads\\files'
ALLOWED_EXTENSIONS = {'txt', 'vec'}
MAX_CONTENT_LENGTH = 250 * 1024 * 1024

app = Flask(__name__)
CORS(app)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

logging.basicConfig(filename="logfiles.log", level=logging.INFO)
print("logging configured")


@app.route('/REST/test', methods=['GET'])
def test():
    return 'CONNECTION WORKS'


@app.route('/REST/uploads/validJSON', methods=['POST'])
def valid_JSON():
    print('Valid JSON called')
    content = request.get_json()
    t1, t2, a1, a2 = JSONFormatter.retrieve_vectors_from_json(content)
    print(t1)
    print(t2)
    print(a1)
    print(a2)
    return 'VALID', 200


# Example: http://127.0.0.1:5000/REST/retrieve_single_vector?embedding_space=fasttext&word=car
@app.route('/REST/vectors/single', methods=['GET'])
def retrieve_single_vector():
    logging.info("APP: Retrieve single vector is called")
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


@app.route('/REST/vectors/multiple', methods=['POST'])
def retrieve_multiple_vectors():
    logging.info("APP: Retrieve multiple vectors is called")
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


@app.route('/REST/augmentations/single', methods=['GET'])
def retrieve_single_augmentation():
    logging.info("APP: Retrieve single augmentation is called")
    bar = request.args.to_dict()
    search = bar['word']
    try:
        augmentations = database_handler.get_augmentation_from_db(search)
        response = jsonify(word=search, augments=[augmentations[i] for i in range(len(augmentations))])
        logging.info("APP: Retrieved vector")
    except:
        return jsonify(message="NOT FOUND"), 404
    return response, 200


@app.route('/REST/augmentations/first10k', methods=['POST'])
def retrieve_multiple_augmentations_10k():
    logging.info("APP: Retrieve single augmentation is called")
    bar = request.args.to_dict()
    space = bar['space']
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


@app.route('/REST/bias-evaluation/all', methods=['POST'])
def bias_evaluations_all():
    logging.info("APP: Bias Evaluation ALL Methods is called")
    content = request.get_json()
    records = request.args.to_dict()
    database = records['space']
    vector_flag = 'false'
    if 'vectors' in records.keys():
        vector_flag = request.args.to_dict()['vectors']
    logging.info("APP: Starting evaluation process")
    if vector_flag == 'false':
        target1, target2, arg1, arg2 = JSONFormatter.retrieve_vectors_evaluation(content, database)
    else:
        target1, target2, arg1, arg2 = JSONFormatter.retrieve_vectors_from_json(content)
    target1, target2 = calculation.check_sizes(target1, target2)
    arg1, arg2 = calculation.check_sizes(arg1, arg2)
    if len(target1) == 0 or len(target2) == 0 or len(arg1) == 0 or len(arg2) == 0:
        logging.info("APP: Stopped, no values found in database")
        return jsonify(message="ERROR: No values found in database."), 404
    logging.info("APP: Final retrieved set sizes: T1=" + str(len(target1)) + " T2=" + str(len(target2)) + " A1=" + str(
        len(arg1)) + " A2=" + str(len(arg2)))
    logging.info("APP: Evaluation process started")
    try:
        result = bias_eval_methods.return_eval_all(target1, target2, arg1, arg2)
    except Exception as e:
        print(e)
        pass
        return jsonify(message="Internal Server Error"), 500
    return result, 200


@app.route('/REST/bias-evaluation/ect', methods=['POST'])
def bias_evaluations_ect():
    logging.info("APP: Bias Evaluation ECT Method is called")
    content = request.get_json()
    database = request.args.to_dict()['space']
    logging.info("APP: Starting evaluation process")
    vector_flag = request.args.to_dict()['vectors']
    if vector_flag == 'false':
        target1, target2, arg1, arg2 = JSONFormatter.retrieve_vectors_evaluation(content, database)
    else:
        target1, target2, arg1, arg2 = JSONFormatter.retrieve_vectors_from_json(content)
    target1, target2 = calculation.check_sizes(target1, target2)
    arg1, arg2 = calculation.check_sizes(arg1, arg2)
    if len(target1) == 0 or len(target2) == 0 or len(arg1) == 0 or len(arg2) == 0:
        logging.info("APP: Stopped, no values found in database")
        return jsonify(message="ERROR: No values found in database."), 404
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
    database = request.args.to_dict()['space']
    logging.info("APP: Starting evaluation process")
    vector_flag = request.args.to_dict()['vectors']
    if vector_flag == 'false':
        target1, target2, arg1, arg2 = JSONFormatter.retrieve_vectors_evaluation(content, database)
    else:
        target1, target2, arg1, arg2 = JSONFormatter.retrieve_vectors_from_json(content)
    target1, target2 = calculation.check_sizes(target1, target2)
    arg1, arg2 = calculation.check_sizes(arg1, arg2)
    if len(target1) == 0 or len(target2) == 0 or len(arg1) == 0 or len(arg2) == 0:
        logging.info("APP: Stopped, no values found in database")
        return jsonify(message="ERROR: No values found in database."), 404
    logging.info("APP: Final retrieved set sizes: T1=" + str(len(target1)) + " T2=" + str(len(target2)) + " A1=" + str(
        len(arg1)) + " A2=" + str(len(arg2)))
    logging.info("APP: Evaluation process started")
    try:
        result = bias_eval_methods.return_eval_bat(target1, target2, arg1, arg2)
    except Exception as e:
        print(e)
        traceback.print_exc()
        return jsonify(message="Internal Server Error"), 500
    return result, 200


@app.route('/REST/bias-evaluation/weat', methods=['POST'])
def bias_evaluations_weat():
    logging.info("APP: Bias Evaluation WEAT Method is called")
    content = request.get_json()
    database = request.args.to_dict()['space']
    logging.info("APP: Starting evaluation process")
    vector_flag = request.args.to_dict()['vectors']
    if vector_flag == 'false':
        target1, target2, arg1, arg2 = JSONFormatter.retrieve_vectors_evaluation(content, database)
    else:
        target1, target2, arg1, arg2 = JSONFormatter.retrieve_vectors_from_json(content)
    target1, target2 = calculation.check_sizes(target1, target2)
    arg1, arg2 = calculation.check_sizes(arg1, arg2)
    if len(target1) == 0 or len(target2) == 0 or len(arg1) == 0 or len(arg2) == 0:
        logging.info("APP: Stopped, no values found in database")
        return jsonify(message="ERROR: No values found in database."), 404
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
    database = request.args.to_dict()['space']
    logging.info("APP: Starting evaluation process")
    vector_flag = request.args.to_dict()['vectors']
    if vector_flag == 'false':
        target1, target2, arg1, arg2 = JSONFormatter.retrieve_vectors_evaluation(content, database)
    else:
        target1, target2, arg1, arg2 = JSONFormatter.retrieve_vectors_from_json(content)
    target1, target2 = calculation.check_sizes(target1, target2)
    if len(target1) == 0 or len(target2) == 0:
        logging.info("APP: Stopped, no values found in database")
        return jsonify(message="ERROR: No values found in database."), 404
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
    content = request.get_json()
    arguments = request.args.to_dict()
    methods = 'gbdd'
    response = debias_methods.return_full_debiasing(methods, arguments, content)
    return response


@app.route('/REST/debiasing/pca/gbdd', methods=['POST'])
def debiasing_pca_gbdd():
    logging.info("APP: Debiasing is called")
    content = request.get_json()
    arguments = request.args.to_dict()
    methods = 'gbdd'
    response = debias_methods.return_pca_debiasing(methods, arguments, content)
    return response


@app.route('/REST/debiasing/full/bam', methods=['POST'])
def debiasing_full_bam():
    logging.info("APP: Debiasing is called")
    # Get content from JSON
    content = request.get_json()
    arguments = request.args.to_dict()
    methods = 'bam'
    response = debias_methods.return_full_debiasing(methods, arguments, content)
    return response


@app.route('/REST/debiasing/pca/bam', methods=['POST'])
def debiasing_pca_bam():
    logging.info("APP: Debiasing is called")
    content = request.get_json()
    arguments = request.args.to_dict()
    methods = 'bam'
    response = debias_methods.return_pca_debiasing(methods, arguments, content)
    return response


@app.route('/REST/debiasing/full/gbddxbam', methods=['POST'])
def debiasing_full_gbdd_bam():
    logging.info("APP: Debiasing is called")
    # Get content from JSON
    content = request.get_json()
    arguments = request.args.to_dict()
    methods = 'gbddxbam'
    response = debias_methods.return_full_debiasing(methods, arguments, content)
    return response


@app.route('/REST/debiasing/pca/gbddxbam', methods=['POST'])
def debiasing_pca_gbdd_bam():
    logging.info("APP: Debiasing is called")
    content = request.get_json()
    arguments = request.args.to_dict()
    methods = 'gbddxbam'
    response = debias_methods.return_pca_debiasing(methods, arguments, content)
    return response


@app.route('/REST/debiasing/full/bamxgbdd', methods=['POST'])
def debiasing_full_bam_gbdd():
    logging.info("APP: Debiasing is called")
    content = request.get_json()
    arguments = request.args.to_dict()
    methods = 'bamxgbdd'
    response = debias_methods.return_full_debiasing(methods, arguments, content)
    return response


@app.route('/REST/debiasing/pca/bamxgbdd', methods=['POST'])
def debiasing_pca_bam_gbdd():
    logging.info("APP: Debiasing is called")
    content = request.get_json()
    arguments = request.args.to_dict()
    methods = 'bamxgbdd'
    response = debias_methods.return_pca_debiasing(methods, arguments, content)
    return response


@app.route('/REST/uploads/embedding-spaces', methods=['POST'])
def upload_embedding_space():
    logging.info("APP: Receiving file from upload")
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
        resp = jsonify({'message': 'Allowed file types are txt, pdf, png, jpg, jpeg, gif'})
        resp.status_code = 400
        print('Case 4')
        return resp


@app.route('/REST/uploads/json/format1', methods=['POST'])
def upload_json_format1():
    return 200


@app.route('/REST/uploads/json/format2', methods=['POST'])
def upload_json_format2():
    return 200


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run()


