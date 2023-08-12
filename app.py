import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and vectorizer
with open('model.pkl', 'rb') as file:
    classifier = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Function to predict the sentiment
def predict_sentiment(tweet):
    text_vectorized = vectorizer.transform([tweet])
    prediction = classifier.predict_proba(text_vectorized)[%]
    return prediction

def main():
    st.set_page_config(page_title="Hate Speech Detection", page_icon="🗣️")
    st.title("Hate Speech Detection")
    tweet = st.text_area("Masukkan tweet:")
    
    if st.button("Prediksi"):
        if tweet.strip() == "":
            st.warning("Masukkan tweet terlebih dahulu!")
        else:
            prediction = predict_sentiment(tweet)
            prediction_hs = prediction[1]*100
            prediction_nhs = prediction[0]*100
            if prediction_hs > prediction_nhs:
                st.success("Hate speech")
            else:
                st.success("Bukan hate speech")
            st.info(f"Prob hate speech: {prediction_hs: .2f}")
            st.info(f"Prob not hate speech: {prediction_nhs: .2f}")

if __name__ == '__main__':
    main()
