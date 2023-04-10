from flask import Flask, request, jsonify
import joblib
import warnings

warnings.filterwarnings("ignore")

# create a Flask application
app = Flask(__name__)

# load the trained machine learning model
model = joblib.load('model.pkl')


# define a route for the "Hello, World!" endpoint
@app.route('/')
def hello():
    return 'Hello, World!'


# define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # get the input data from the user
    data = request.get_json()

    # assign input to each variable
    input_data = [
        data['Age'],
        data['Gender'],
        data['Ethnicity'],
        data['Religion'],
        data['Marital_Status'],
        data['Employment'],
        data['Little_Interest'],
        data['Feeling_Down'],
        data['Sleeping_Trouble'],
        data['Feeling_Tired'],
        data['Poor_Appetite'],
        data['Feeling_Bad'],
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

    # use the loaded model to make predictions
    predictions = model.predict(predict_data)

    # prediction result
    result = str(predictions[0])

    if result == '0':
        return jsonify({'predictions': "Negative"})
    else:
        return jsonify({'predictions': "Positive"})


# define a function that starts the Flask application
def run():
    app.run(debug=True)


if __name__ == '__main__':
    run()
