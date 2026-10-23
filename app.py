
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    layout="centered"
)

@st.cache_resource
def load_resources():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    model = joblib.load("emotion_detection_model.pkl")
    return model

model = load_resources()

stop_words = set(stopwords.words("english"))
stop_words = stop_words - {"no", "not", "nor"}
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

st.title("😊 Emotion Detection Using NLP")

st.write(
    "Enter a sentence to predict the emotion "
    "expressed in the text."
)

user_input = st.text_area(
    "Enter your text:",
    placeholder="I am feeling very happy today..."
)

emotion_emojis = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲"
}

if st.button("Predict Emotion"):
    if user_input.strip():
        cleaned_text = preprocess_text(user_input)

        if cleaned_text:
            prediction = model.predict([cleaned_text])[0]
            emoji = emotion_emojis.get(prediction, "🙂")

            st.success(
                f"Predicted Emotion: {emoji} "
                f"{prediction.capitalize()}"
            )
        else:
            st.warning("Please enter meaningful text.")
    else:
        st.warning("Please enter some text.")
