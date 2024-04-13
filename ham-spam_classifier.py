from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

svc_model = joblib.load('spam_ham_svc_model_countvectorizer.pkl')
count_vectorizer = joblib.load('count_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']

        message_vectorized = count_vectorizer.transform([message])
        
        prediction = svc_model.predict(message_vectorized)[0]
        if prediction == 1:
            result = 'Spam'
        else:
            result = 'Ham'
        return render_template('result.html', message=message, result=result)

if __name__ == '__main__':
    app.run(debug=True)