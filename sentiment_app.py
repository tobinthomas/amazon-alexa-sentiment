import pickle
import re

import numpy as np
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load pre-trained models and vectorizer
cv = pickle.load(open('Models/countVectorizer.pkl', 'rb'))
scaler = pickle.load(open('Models/scaler.pkl', 'rb'))
model_xgb = pickle.load(open('Models/model_xgb.pkl', 'rb'))  # Using XGBoost Model

# Initialize PorterStemmer
stemmer = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

# Function to clean and preprocess text
def preprocess_review(review):
    review = re.sub('[^a-zA-Z]', ' ', review)  # Remove special characters
    review = review.lower().split()  # Convert to lowercase and split
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]  # Stemming and stopword removal
    return ' '.join(review)

# Streamlit App UI
st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")
st.title("Amazon Review Sentiment Analysis")

# Text Input
review_text = st.text_area("Enter your review:", height=150)

# Predict Button
if st.button("Predict Sentiment"):
    if review_text.strip():
        # Preprocess the text
        processed_review = preprocess_review(review_text)
        transformed_text = cv.transform([processed_review]).toarray()  # Convert text to numerical features
        scaled_text = scaler.transform(transformed_text)  # Scale the features
        
        # Make Prediction
        prediction = model_xgb.predict(scaled_text)[0]

        # Display Sentiment
        if prediction == 1:
            st.success(" Positive Sentiment ")
        else:
            st.error(" Negative Sentiment ")
    else:
        st.warning(" Please enter a review before predicting.")

# Footer
st.markdown("---")
st.markdown("Developed with  using Streamlit")

