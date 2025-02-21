import pickle
from flask import Flask, render_template, request

# Load the trained model and vectorizer from the 'models' folder
with open('models/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('models/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_authenticity', methods=['POST'])
def check_authenticity():
    # Get the text entered by the user
    content = request.form['content']
    
    # Transform the input content using the same vectorizer used during training
    content_vectorized = vectorizer.transform([content])
    
    # Predict the class (REAL or FAKE)
    prediction = model.predict(content_vectorized)
    
    # Return the appropriate response based on the prediction
    if prediction[0] == 'REAL':
        result = "The news is REAL."
    else:
        result = "The news is FAKE."

    # Pass the result and article content to the template
    return render_template('index.html', result=result, article_content=content)

if __name__ == '__main__':
    app.run(debug=True)
