# 📧 EMAIL SPAM DETECTION

import kagglehub
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Step 1: Download Dataset

print("Downloading dataset...")
path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
print("Dataset downloaded at:", path)

# Step 2: Load CSV File

files = os.listdir(path)

for file in files:
    if file.endswith(".csv"):
        csv_file = os.path.join(path, file)
        break

print("CSV File Found:", csv_file)

df = pd.read_csv(csv_file, encoding="latin-1")

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 3: Train-Test Split

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Convert Text to Numerical (CountVectorizer)

vectorizer = CountVectorizer(stop_words='english')

X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# Step 5: Train Model (Naive Bayes)

model = MultinomialNB()
model.fit(X_train_vector, y_train)

# Step 6: Evaluate Model

y_pred = model.predict(X_test_vector)

print("\n==============================")
print("📊 MODEL PERFORMANCE")
print("==============================")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Custom Prediction

def predict_email(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return "🚨 Spam" if prediction[0] == 1 else "✅ Not Spam"


print("\n==============================")
print("📩 CUSTOM PREDICTIONS")
print("==============================")

test1 = "Congratulations! You won a free lottery ticket."
test2 = "URGENT! Claim your prize now."
test3 = "Hey, are we meeting tomorrow?"

print("\nMessage:", test1)
print("Prediction:", predict_email(test1))

print("\nMessage:", test2)
print("Prediction:", predict_email(test2))

print("\nMessage:", test3)
print("Prediction:", predict_email(test3))
