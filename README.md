# Email & SMS Spam Classifier  
_End-to-End NLP & Machine Learning Project_

A machine learning–based web application that classifies Email/SMS messages as **Spam** or **Not Spam (Ham)** using **Natural Language Processing (NLP)** techniques and **classical ML models**.  
The model is deployed using **Streamlit** for real-time, interactive predictions.

---

## 🚀 Project Overview

Spam messages pose security and productivity risks.  
This project aims to automatically detect spam messages by learning patterns from labeled text data using NLP and ML.

The project follows a **complete ML lifecycle**:
- Data preprocessing
- Feature extraction
- Model training & evaluation
- Model persistence
- Deployment using Streamlit

---

## 🛠 Tech Stack

### Programming Language
- Python

### Libraries & Frameworks
- Streamlit
- Scikit-learn
- NLTK
- Pandas
- NumPy
- Pickle

---

## 📂 Project Structure
Email_SMS_spam_classifier/
│
├── app.py # Streamlit web application
├── model.pkl # Trained ML model
├── vectorizer.pkl # TF-IDF vectorizer
├── spam_detection.ipynb # Training & experimentation notebook
├── requirements.txt # Project dependencies
└── README.md # Project documentation

---

## ⚙️ Workflow

### 1. Data Preprocessing (NLP)
- Lowercasing text
- Tokenization
- Removal of special characters
- Stopword removal
- Stemming using Porter Stemmer

### 2. Feature Engineering
- TF-IDF Vectorization to convert text into numerical features

### 3. Model Training
- Machine learning classifier trained on vectorized text
- Model evaluation using accuracy and classification metrics

### 4. Model Persistence
- Trained model and vectorizer saved using `pickle`
- Reused for inference in the Streamlit app

### 5. Deployment
- Streamlit-based UI for real-time prediction
- Displays classification result and confidence score

---
