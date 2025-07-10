from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    question = request.form['question']
    question_tfidf = vectorizer.transform([question])
    prediction = model.predict(question_tfidf)[0]
    return render_template('index.html', prediction=prediction, question=question)

if __name__ == '__main__':
    app.run(debug=True)
