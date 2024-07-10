from flask import Flask, render_template, request 
import pickle

app = Flask(__name__)
model = pickle.load(open('savedmodel1.sav', 'rb'))

@app.route('/')
def home():
    result = ''
    return render_template('index.html', **locals())

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    SepalLengthCm = request.form.get('sepal_length')
    sepal_width = request.form.get('sepal_width')
    petal_length = request.form.get('petal_length')
    petal_width = request.form.get('petal_width')
    result = model.predict([[SepalLengthCm, sepal_width, petal_length, petal_width]])[0]
    return render_template('index.html', **locals())

if __name__ == '__main__':
    app.run(debug=True)