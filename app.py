import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from newspaper import Article
from urllib.parse import urlparse
import pdfplumber
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ==================================================
# CONFIG
# ==================================================

st.set_page_config(
    page_title="NewsGuard AI",
    page_icon="📰",
    layout="wide"
)

MAX_LEN = 500

# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
    "models/bigru_attention_model.keras",
    compile=False
)

    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


model, tokenizer = load_model()

# ==================================================
# SOURCE CREDIBILITY
# ==================================================

SOURCE_SCORES = {
    "bbc.com": 95,
    "reuters.com": 98,
    "cnn.com": 85,
    "foxnews.com": 80,
    "nytimes.com": 96,
    "theguardian.com": 94
}

# ==================================================
# HELPERS
# ==================================================

def predict_article(text):

    seq = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(
        padded,
        verbose=0
    )[0][0]

    confidence = round(
        max(prediction, 1 - prediction) * 100,
        2
    )

    label = (
        "FAKE NEWS"
        if prediction > 0.5
        else "REAL NEWS"
    )

    return label, confidence


def extract_news(url):

    try:

        article = Article(url)

        article.download()
        article.parse()

        return {
            "title": article.title,
            "authors": article.authors,
            "text": article.text,
            "date": article.publish_date
        }

    except Exception as e:

        st.error(f"Extraction Error: {e}")

        return None


def display_prediction(label, confidence):

    if label == "FAKE NEWS":
        st.error(f"🚨 {label}")
    else:
        st.success(f"✅ {label}")

    st.metric("Confidence", f"{confidence}%")
    st.progress(int(confidence))

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("📰 NewsGuard AI")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Fake News Detection"
    ]
)

# ==================================================
# FAKE NEWS DETECTION
# ==================================================

if menu == "Fake News Detection":

    st.title("📰 NewsGuard AI")
    st.write("Analyze news articles using Deep Learning.")

    input_mode = st.radio(
        "Choose Input Type",
        [
            "Paste Text",
            "News URL",
            "Upload File"
        ]
    )

    # ------------------------
    # PASTE TEXT
    # ------------------------

    if input_mode == "Paste Text":

        article = st.text_area(
            "Paste News Article",
            height=300
        )

        if st.button("Analyze Text"):

            if article.strip():

                with st.spinner("Analyzing..."):

                    label, confidence = predict_article(article)

                display_prediction(
                    label,
                    confidence
                )

            else:

                st.warning("Please enter some text.")

    # ------------------------
    # NEWS URL
    # ------------------------

    elif input_mode == "News URL":

        url = st.text_input(
            "Enter News URL"
        )

        if st.button("Extract & Analyze"):

            if url.strip():

                with st.spinner("Extracting article..."):

                    data = extract_news(url)

                if data:

                    st.subheader(data["title"])

                    st.write("Authors:", data["authors"])

                    st.write("Published:", data["date"])

                    st.text_area(
                        "Extracted Article",
                        data["text"][:5000],
                        height=300
                    )

                    with st.spinner("Analyzing..."):

                        label, confidence = predict_article(
                            data["text"]
                        )

                    display_prediction(
                        label,
                        confidence
                    )

                    domain = urlparse(url).netloc.replace(
                        "www.",
                        ""
                    )

                    score = SOURCE_SCORES.get(
                        domain,
                        50
                    )

                    st.metric(
                        "Source Credibility",
                        f"{score}/100"
                    )

            else:

                st.warning("Please enter a valid URL.")
    # ------------------------
    # FILE UPLOAD
    # ------------------------

    elif input_mode == "Upload File":

        uploaded_file = st.file_uploader(
            "Upload TXT or PDF",
            type=["txt", "pdf"]
        )

        if uploaded_file is not None:

            text = ""

            if uploaded_file.name.lower().endswith(".txt"):

                text = uploaded_file.read().decode(
                    "utf-8",
                    errors="ignore"
                )

            elif uploaded_file.name.lower().endswith(".pdf"):

                with st.spinner("Reading PDF..."):

                    with pdfplumber.open(uploaded_file) as pdf:

                        for page in pdf.pages:

                            page_text = page.extract_text()

                            if page_text:
                                text += page_text + "\n"

            if text.strip():

                st.text_area(
                    "Extracted Content",
                    text[:5000],
                    height=300
                )

                if st.button("Analyze File"):

                    with st.spinner("Analyzing..."):

                        label, confidence = predict_article(text)

                    display_prediction(
                        label,
                        confidence
                    )

            else:

                st.warning(
                    "No readable text found in the uploaded file."
                )
