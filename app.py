import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Streamlit Page Config
st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .title {
        font-size: 38px !important;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
    }
    .result {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    }
    .spam {
        background-color: #ffcccc;
        color: #b30000;
    }
    .not-spam {
        background-color: #ccffcc;
        color: #006600;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown('<p class="title">📩 Email/SMS Spam Classifier</p>', unsafe_allow_html=True)

# User Input
input_sms = st.text_area("✉️ Enter your message below:", placeholder="Type your message here...")

if st.button("🚀 Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message before prediction.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.markdown('<div class="result spam">🚨 This message is Spam!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result not-spam">✅ This message is Not Spam!</div>', unsafe_allow_html=True)
