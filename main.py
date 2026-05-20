import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model


word_index = imdb.get_word_index()
reversed_word_index = {value: key for key, value in word_index.items()}

model = load_model('imbd_rnn_model.h5')


def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# funciton to accept user input and make prediction
def preprocess_text(review):
    words = review.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review



def predict_review_sentiment(review):
    
    # preprocess the review
    preprocessed_review = preprocess_text(review)

    # prediction
    prediction = model.predict(preprocessed_review)

    # sentiment decision
    sentiment = 'positive' if prediction[0][0] >= 0.5 else 'negative'

    return sentiment, prediction[0][0]



import streamlit as st

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive or negative).")

user_input = st.text_area("Movie Review", "")

if st.button("Predict Sentiment"):
    if user_input:
        preprocess_text = preprocess_text(user_input)
        prediction = model.predict(preprocess_text)
        sentiment = 'positive' if prediction[0][0] >= 0.5 else 'negative'
        st.write(f"Predicted Sentiment: {sentiment} (Score: {prediction[0][0]:.4f})")
    else:
        st.write("Please enter a movie review to predict its sentiment.")
