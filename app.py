
import streamlit as st
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="AI SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0B1220;
        color: white;
    }

    .main-title {
        text-align: center;
        color: #22D3EE;
        font-size: 36px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        color: #CBD5E1;
        font-size: 16px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
    }

    .spam {
        background-color: #451A1A;
        border: 1px solid #EF4444;
        color: #FCA5A5;
    }

    .ham {
        background-color: #123524;
        border: 1px solid #22C55E;
        color: #86EFAC;
    }
</style>
""", unsafe_allow_html=True)


# Load trained model
@st.cache_resource
def load_model():
    if os.path.exists("spam_model.pkl"):
        return joblib.load("spam_model.pkl")
    return None


model = load_model()


# Application heading
st.markdown(
    '<div class="main-title">📩 AI SMS Spam Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent SMS Classification using Machine Learning and NLP'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

st.subheader("🔍 Analyze Your SMS Message")

st.write(
    "Enter a message below to determine whether it is Spam "
    "or a Legitimate message."
)

# SMS input
message = st.text_area(
    "Enter your SMS message",
    placeholder="Example: Congratulations! You have won a free prize...",
    height=130
)

# Prediction
if st.button("🚀 Detect Spam", use_container_width=True):

    if not message.strip():
        st.warning("Please enter an SMS message.")

    elif model is None:
        st.error(
            "Trained model not found. Please run train_model.py "
            "first to download the dataset and train the model."
        )

    else:
        prediction = model.predict([message])[0]

        probabilities = model.predict_proba([message])[0]
        confidence = max(probabilities) * 100

        if prediction == 1:
            st.markdown(
                """
                <div class="result-box spam">
                    <h2>🚨 SPAM MESSAGE</h2>
                    <p>This message is classified as Spam.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                """
                <div class="result-box ham">
                    <h2>✅ LEGITIMATE MESSAGE</h2>
                    <p>This message is classified as Ham.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

st.divider()

# Example messages
st.subheader("💡 Try Example Messages")

st.write("**Example 1 – Spam:**")
st.code(
    "Congratulations! You have won a free prize. "
    "Call now to claim your reward."
)

st.write("**Example 2 – Legitimate:**")
st.code(
    "Hey, are we meeting today at 5 PM?"
)

st.divider()

# Project information
st.subheader("ℹ️ About This Project")

st.write("""
This application uses Natural Language Processing (NLP)
and Machine Learning to classify SMS messages.

**Machine Learning Algorithm:** Logistic Regression

**Feature Extraction:** TF-IDF Vectorization

**Framework:** Scikit-learn and Streamlit

**Classification Categories:**
- Spam
- Legitimate (Ham)
""")

st.caption(
    "AI-Powered SMS Spam Detection | Python | NLP | Machine Learning"
)
