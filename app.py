from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os
import joblib
import warnings

warnings.filterwarnings("ignore")

# create a Flask application
app = Flask(__name__)
CORS(app)
env_config = os.getenv("PROD_APP_SETTINGS", "config.DevelopmentConfig")
app.config.from_object(env_config)

# define a route for the "Hello, World!" endpoint
@app.route('/')
@cross_origin()
def index():
    return 'This is ML Model used for SBDS.'


# define a route for making predictions
@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    # get the input data from the user
    data = request.get_json()

    # assign input to each variable
    algo = data['Algo']
    input_data = [
        data['Age'],
        data['Gender'],
        data['Ethnicity'],
        data['Employment'],
        data['Little_Interest'],
        data['Feeling_Down'],
        data['Sleeping_Trouble'],
        data['Feeling_Tired'],
        data['Trouble_Concentrating'],
        data['Moving_Slowly'],
        data['Thoughts_self_harm'],
        data['Has_Depressive_Disorder'],
        data['Past_Psychiatric_Disorder'],
        data['Past_Suicidal_Attempt'],
        data['Medical_Comorbidity']
    ]

    # create data format for prediction
    predict_data = [input_data]

    # load the trained machine learning model and algorithm name
    if algo == 'dtree':
        algo_name = 'Decision Tree'
        model = joblib.load('model_dtree.pkl')
    elif algo == 'nb':
        algo_name = 'Naive Bayes'
        model = joblib.load('model_nb.pkl')
    else:
        return jsonify({
            'algorithm': "",
            'prediction': "",
            'error': "No algorithm method is provided!"
        })

    # make prediction
    predictions = model.predict(predict_data)

    # prediction result
    result = str(predictions[0])

    if result == '0':
        return jsonify({
            'algorithm': algo_name,
            'prediction': "Low Risk"
        })
    else:
        return jsonify({
            'algorithm': algo_name,
            'prediction': "High Risk"
        })
