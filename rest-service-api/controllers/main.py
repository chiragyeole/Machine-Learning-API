from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import os, io, csv, datetime, uuid, pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'exp.sqlite')
db = SQLAlchemy(app)
ma = Marshmallow(app)

# Initialize database model & create schema
class Experiment(db.Model):
    id = db.Column(db.String(100), primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    start_date = db.Column(db.DateTime, default=datetime.datetime.now(), nullable=False)
    type = db.Column(db.String(255), nullable=False)
    train_data = db.Column(db.String(500), nullable=False)
    test_data = db.Column(db.String(500), nullable=False)
    model = db.Column(db.String(255))
    result = db.Column(db.String(120))

    def __init__(self, id, name, start_date, type, train_data, test_data, model, result):
        self.id = id
        self.name = name
        self.start_date = datetime.datetime.now()
        self.type = type
        self.train_data = train_data
        self.test_data = test_data
        self.model = model
        self.result = result

class ExperimentSchema(ma.Schema):
    # Fields to expose
    class Meta:
        fields = ('id', 'name', 'start_date', 'type', 'train_data', 'test_data', 'model', 'result')


exp_schema = ExperimentSchema()
exps_schema = ExperimentSchema(many=True)

# endpoint to create new experiment
@app.route('/experiment', methods=['POST'], endpoint='create_experiment')
def create_handler():
    response = {}
    if request.method == 'POST':
        input = request.get_json()
        response["message"], response["code"] = add_experiment(input)

    return jsonify(response)

def add_experiment(input):
    try:
        exp = Experiment.query.get(input['id'])
        if exp is not None:
            msg, status_code = "Experiment with id: "+ exp.id + " already exists.", 400
            return msg, status_code

        id = input['id']
        name = input['name']
        start_date = datetime.datetime.now()
        type = input['type']
        train_data = input['train_data']
        test_data = input['test_data']
        model = input['model']
        result = input['result']

        new_exp = Experiment(id, name, start_date, type, train_data, test_data, model, result)
        db.session.add(new_exp)
        db.session.commit()
        msg, status_code = "Experiment created with id: " + new_exp.id + " successfully!!!", 201

    except Exception as e:
        print(e)
        msg, status_code = "Failed to create Experiment with id: " + new_exp.id + ".", 400
    finally:
        return msg, status_code

# endpoint to show all experiments
@app.route('/experiment', methods=['GET'], endpoint='all_experiments')
def all_experiments():
    response = {}
    try:
        all_exps = Experiment.query.all()
        output = exps_schema.dump(all_exps)
        response['success'], response['status_code'], response['data'] = str(len(output.data)) + " experiments retrieved successfully!!!", 200, output.data
    except Exception as e:
        print(e)
        response['error'], response['status_code'], response['data'] = "Failed to retrive Experiments.", 400
    finally:
        return jsonify(response)

# endpoint to get experiment detail by id
@app.route('/experiment/<id>', methods=['GET'], endpoint='retrieve_experiment')
def retrieve_experiment(id):
    response = {}
    try:
        exp = Experiment.query.get(id)
        if exp is None:
            response['error'], response['status_code'] = "No Experiment found for the id: "+ str(id), 404
            return jsonify(response)
            
        output = exp_schema.dump(exp)
        response['success'], response['status_code'], response['data'] = "Experiment with id: "+ exp.id +" retrieved successfully!!!", 200, output.data
    except Exception as e:
        print(e)
        response['error'], response['status_code'], response['data'] = "Failed to retrive Experiment with id: "+ exp.id + ".", 404
    finally:
        return jsonify(response)

# endpoint to update details of experiment given in id
@app.route('/experiment/<id>', methods=['PUT'], endpoint='update_experiment')
def update_experiment(id):
    response = {}
    try:
        exp = Experiment.query.get(id)
        if exp is None:
            response['error'], response['status_code'] = "No Experiment found for the id: "+ str(id), 404
            return jsonify(response)

        input = request.get_json()
        exp.name = input['name']
        exp.start_date = datetime.datetime.now()
        exp.type = input['type']
        exp.train_data = input['train_data']
        exp.test_data = input['test_data']
        exp.model = input['model']
        exp.result = input['result']
        db.session.commit()
        response['success'], response['status_code'] = "Experiment with id: "+ exp.id +" updated successfully!!!", 200
    except Exception as e:
        print(e)
        response['error'], response['status_code'], response['data'] = "Failed to update the Experiment with id: "+ exp.id + ".", 400
    finally:
        return jsonify(response)

# endpoint to delete experiment by id
@app.route('/experiment/<id>', methods=['DELETE'], endpoint='delete_experiment')
def delete_experiment(id):
    response = {}
    try:
        exp = Experiment.query.get(id)
        if exp is None:
            response['error'], response['status_code'] = "No Experiment found for the id: "+ str(id), 404
            return jsonify(response)
        db.session.delete(exp)
        db.session.commit()
        response['success'], response['status_code'] = "Experiment with id: "+ id +" deleted successfully!!!", 204
    except Exception as e:
        print(e)
        response['error'], response['status_code'] = "Failed to delete the Experiment with id: "+ id + ".", 400
    finally:
        return jsonify(response)

# currently allowing only csv files for dataset
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# endpoind to upload the dataset from the api
@app.route('/upload', methods=['POST'], endpoint='upload_dataset')
def upload_dataset():
    response = {}
    try:
        UPLOAD_FOLDER = os.path.abspath('data') + '/'
        input = request.get_json()
        filepath = UPLOAD_FOLDER + input['file']
        if not os.path.isfile(filepath):
            response['messsage'], response['status_code'] = "No file found", 404
            return jsonify(response)
        filename = os.path.basename(filepath).split('.')[0].lower()
        if allowed_file(input['file']):
            df = pd.read_csv(filepath)

            msk = np.random.rand(len(df)) <= 0.7
            train = df[msk]
            test = df[~msk]
 
            train_file_path = UPLOAD_FOLDER + filename + '_train.csv'
            test_file_path = UPLOAD_FOLDER + filename + '_test.csv'
            train.to_csv(train_file_path, index=False)
            test.to_csv(test_file_path, index=False)

            exp = {
            'id' : input['id'],
            'name': filename,
            'start_date': datetime.datetime.now(),
            'type': input['type'],
            'train_data': train_file_path,
            'test_data': test_file_path,
            'result': 'N/A',
            'model': 'N/A'
            }

            msg, status_code = add_experiment(exp)  
            response['message'], response['status_code'] = msg, status_code
        else:
            response['message'], response['status_code'], response['data'] = "The file format allowed is only csv.", 400
    except Exception as e:
        print(e)
        response['message'], response['status_code'], response['data'] = "bad request", 400
    finally:
        return jsonify(response)

# endpoind to train your model on the dataset denoted by id
@app.route('/train/<id>', methods=['POST'])
def train(id):
    response = {}
    try:
        UPLOAD_FOLDER = os.path.abspath("data")+"/"

        exp = Experiment.query.get(id)
        if exp is None:
            response['error'], response['status_code'] = "No Experiment found for the id: "+ str(id), 400
            return jsonify(response)

        output = exp_schema.dump(exp)
        input = output.data

        data = pd.read_csv(input['train_data'])
        cols = data.shape[1]
        X_train = data.iloc[:, 0:cols - 1]
        y_train = data.iloc[:, cols - 1]

        model = RandomForestClassifier(n_estimators=10, n_jobs=4)
        model.fit(X_train, y_train)

        pklfilename = UPLOAD_FOLDER + input['name'] + ".pkl"
        pickle.dump(model, open(pklfilename, "wb"))


        exp.start_date = datetime.datetime.now()
        exp.model = pklfilename
        db.session.commit()

        response['success'], response['status_code'] = message, status_code = "Model trained and stored successfully!!!", 200

    except Exception as e:
        print(e)
        response['error'], response['status_code'] = "Model couldn't be trained or stored.", 400
    finally:
        return jsonify(response);

# endpoind to test your model on the dataset denoted by id
@app.route('/test/<id>', methods=['POST'])
def test(id):
    response = {}
    try:
        exp = Experiment.query.get(id)
        if exp is None:
            response['error'], response['status_code'] = "No Experiment found for the id: "+ str(id), 400
            return jsonify(response)

        output = exp_schema.dump(exp)
        input = output.data

        model = pickle.load(open(input['model'], "rb"))
        if model is None:
            rresponse['error'], response['status_code'] = "Model file not found", 400

        data = pd.read_csv(input['test_data'])
        cols = data.shape[1]
        X_test = data.iloc[:, 0:cols - 1]
        y_test = data.iloc[:, cols - 1]

        predictions = model.predict(X_test)
        score =  model.score(X_test, y_test)
        f1score = f1_score(y_test, predictions, average='weighted')

        result = {"accuracy_score": score, "f1_score": f1score}

        exp.start_date = datetime.datetime.now()
        exp.result = str(result)
        db.session.commit();

        response['success'], response['status_code'], response['result'] = "Model tested  successfully. Check out the result", 200, result
    except Exception as e:
        print(e)
        response['error'], response['status_code'] = "Model couldn't be tested", 400
    finally:
        return jsonify(response)

# endpoind to predict classes on the dataset denoted by id using given data
@app.route('/predict/<id>', methods=['POST'])
def predict(id):
    response = {}
    try:
        input = request.get_json()
        exp = Experiment.query.get(id)
        if exp is None:
            response['error'], response['status_code'] = "No Experiment found for the id: "+ str(id), 400
            return jsonify(response)

        output = exp_schema.dump(exp)
        row = output.data

        model = pickle.load(open(row['model'], "rb"))
        data = pd.read_csv(row['test_data'])
        cols = data.shape[1]
        columns = set(data.columns.values[0: cols - 1])
        input_columns = set(map(str,input.keys()))
        if columns.issubset(input_columns) is False:
            response['error'], response['status_code'] = "Bad request please check all the columns in the input data it should match the training data columns", 400
            return jsonify(response)

        predict_data = {}
        for key, value in input.items():
            if str(key) in columns:
                predict_data[key] = value

        predict_array = np.array(list(predict_data.values())).astype(float)
        predictions = model.predict([predict_array])
   
        response['success'], response['status_code'], response['predicted_class'] = "Prediction complete. Check out the predictions", 200, str(predictions[0])
    except Exception as e:
        print(e, 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
        response['error'], response['status_code'] = "Could not predict based on the input data", 40
        
    finally:
        return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
    app.run(port = 5000, host = '0.0.0.0', threaded=True)