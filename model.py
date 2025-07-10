import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Try to load AI.parquet first, then fallback to AI.csv
if os.path.exists("AI.parquet"):
    df = pd.read_parquet("AI.parquet")
    print("Loaded data from AI.parquet")
elif os.path.exists("AI.csv"):
    df = pd.read_csv("AI.csv")
    print("Loaded data from AI.csv")
else:
    raise FileNotFoundError("Neither AI.parquet nor AI.csv found.")

# Drop rows with missing values in Question or Answer
df = df.dropna(subset=['Question', 'Answer'])

# Features and Target
X = df['Question']
y = df['Answer']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# Evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))