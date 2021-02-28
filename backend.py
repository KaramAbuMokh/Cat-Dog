# Import Required Libraries

from flask import Flask, render_template, request, send_file, send_from_directory, jsonify
import pickle
import numpy as np

# We need to initialise the Flask object to run the flask app
# By assigning parameters as static folder name,templates folder name

app = Flask(__name__, static_folder='static', template_folder='templates')

# We need to load the pickled model file so as to use it for the prediction

model = pickle.load(open('model.pkl', 'rb+'))


# For the root '/' we need to define a function in which we are rendering the template of index.html as default
# This rendering template is done if it get's any GET Request

@app.route('/', methods=['POST', 'GET'])
def main():
    if request.method == 'GET':
        return render_template('index.html')


# For the root '/predict' we need to define a function named predict
# This function will take values from the ajax request and performs the prediction
# By getting response from flask to ajax it will display the response to the result field
# This whole above process occurs when request method is POST
# This rendering template is index.html if it get's any GET Request

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        # Converting all the form values to float and making them append in a list(features)
        features = [float(x) for x in request.form.values()]
        # Printing the features for debug purpose
        print(features)
        # Predicting the label for the features collected
        labels = model.predict([features])
        # Printing the labels array for debug purpose
        print(labels)
        # Storing the result from the labels array
        species = labels[0]
        # If species is 0 species is setosa else if species is 1 then species is VersiColor else it is Virginica
        if species == 0:
            s = "It is Iris Setosa"
        elif species == 1:
            s = "It is Iris VersiColor"
        else:
            s = "It is Iris Virginica"
        # Returning the response to ajax
        return s


# It is the starting point of code
if __name__ == '__main__':
    # We need to run the app to run the server
    app.run(debug=False)