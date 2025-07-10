from flask import Flask, render_template, request, jsonify
import pickle
import os

app = Flask(__name__)

# Load model and vectorizer with error handling
try:
    if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
        model = pickle.load(open("model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
        print("✅ Model and vectorizer loaded successfully!")
    else:
        print("❌ Model files not found. Please run model.py first.")
        model = None
        vectorizer = None
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model = None
    vectorizer = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        return render_template('index.html', 
                             error="Model not loaded. Please contact administrator.",
                             question=request.form.get('question', ''))
    
    try:
        question = request.form.get('question', '').strip()
        
        if not question:
            return render_template('index.html', 
                                 error="Please enter a question.",
                                 question=question)
        
        # Make prediction
        question_tfidf = vectorizer.transform([question])
        prediction = model.predict(question_tfidf)[0]
        
        return render_template('index.html', 
                             prediction=prediction, 
                             question=question)
    
    except Exception as e:
        return render_template('index.html', 
                             error=f"Prediction error: {str(e)}", 
                             question=request.form.get('question', ''))

# API endpoint for AJAX requests (optional)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not model or not vectorizer:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        question_tfidf = vectorizer.transform([question])
        prediction = model.predict(question_tfidf)[0]
        
        return jsonify({
            'prediction': prediction,
            'question': question,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)