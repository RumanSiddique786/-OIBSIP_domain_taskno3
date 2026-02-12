# -OIBSIP_domain_taskno3
this is my third task as a datascience intern

#  Email Spam Detection using Machine Learning

##  Project Overview

Spam emails (junk mails) are unsolicited messages sent in bulk, often containing advertisements, scams, or phishing attempts. Detecting spam automatically is essential for email security and user protection.

This project builds a **Machine Learning-based Email Spam Detection System** using Python.  
The model classifies messages into:

-  Not Spam (Ham)
-  Spam

---

##  Objective

The objective of this project is to:

- Build an email spam classifier using Machine Learning
- Convert text data into numerical format
- Train a model to detect spam messages accurately
- Evaluate model performance using standard metrics
- Test custom user input messages

---

##  Dataset Used

- **Dataset:** UCI SMS Spam Collection Dataset  
- Downloaded using: `kagglehub`
- Contains labeled SMS messages:
  - `ham` → Not Spam
  - `spam` → Spam

---

##  Tools & Technologies Used

- Python
- Pandas
- Scikit-learn
- KaggleHub
- CountVectorizer
- Multinomial Naive Bayes

---

##  Steps Performed

### 1️⃣ Dataset Download
- Used KaggleHub to download the SMS Spam Collection dataset.

### 2️⃣ Data Preprocessing
- Selected relevant columns
- Renamed columns to `label` and `message`
- Converted labels:
  - ham → 0
  - spam → 1

### 3️⃣ Train-Test Split
- Split dataset into:
  - 80% Training Data
  - 20% Testing Data

### 4️⃣ Text Vectorization
- Used **CountVectorizer**
- Converted text messages into numerical feature vectors
- Removed English stopwords

### 5️⃣ Model Training
- Used **Multinomial Naive Bayes**
- Suitable for text classification problems

### 6️⃣ Model Evaluation
Evaluated using:
- Accuracy Score
- Confusion Matrix
- Precision
- Recall
- F1-Score

### 7️⃣ Custom Message Testing
- Tested user-defined messages
- Model predicts Spam or Not Spam

---

##  Model Performance

-  Accuracy: **98%**
-  Spam Detection Recall: **92%**
-  Very low false positives and false negatives

Example Predictions:

| Message | Prediction |
|----------|------------|
| "Congratulations! You won a free lottery ticket." | 🚨 Spam |
| "URGENT! Claim your prize now." | 🚨 Spam |
| "Hey, are we meeting tomorrow?" | ✅ Not Spam |

---



