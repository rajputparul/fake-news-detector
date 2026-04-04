from flask import Flask, request, jsonify
from flask import render_template
import pickle
app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']

    vector = vectorizer.transform([news])
    prediction = model.predict(vector)[0]

    if prediction == 1:
       result = "<span style='color:green'>Real News </span>"
    else:
       result = "<span style='color:red'>Fake News </span>"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)