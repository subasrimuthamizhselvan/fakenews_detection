# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Function to train the model
def train_model():
    # Load the dataset
    df = pd.read_csv('data/news.csv')  # Correct the path to news.csv inside the 'data' folder

    # Display the shape of the dataset and the first few rows for verification
    print(f"Dataset Shape: {df.shape}")
    print(df.head())

    # Data Preprocessing
    # Assuming the columns are 'title', 'text', and 'label'
    # Combine 'title' and 'text' for model training
    df['content'] = df['title'] + " " + df['text']
    X = df['content']  # Features
    y = df['label']    # Labels

    # Text vectorization using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(X)

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model: Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Save the trained model and vectorizer
    with open('models/model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('models/vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    print("Model and vectorizer have been saved successfully.")

# Run the training function
if __name__ == "__main__":
    train_model()
