
import os
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from train_model import train_model

# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="AI SMS Spam Detection",
    page_icon="📩",
    layout="wide"
)

# ----------------------------------
# CUSTOM CSS
# ----------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0B1220;
    color: #FFFFFF;
}

.main-title {
    color: #22D3EE;
    font-size: 38px;
    font-weight: bold;
}

.subtitle {
    color: #94A3B8;
    font-size: 16px;
}

.result-box {
    padding: 22px;
    border-radius: 12px;
    text-align: center;
    margin: 15px 0;
}

.spam {
    background: #451A1A;
    border: 1px solid #EF4444;
}

.ham {
    background: #123524;
    border: 1px solid #22C55E;
}

div[data-testid="stMetric"] {
    background: #172337;
    padding: 15px;
    border-radius: 10px;
}

.stButton button {
    background-color: #0891B2;
    color: white;
    border-radius: 8px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ----------------------------------
# MODEL LOADING
# ----------------------------------

MODEL_PATH = "models/spam_model.pkl"
EVALUATION_PATH = "models/evaluation.pkl"


@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        with st.spinner("Training AI model for first-time use..."):
            train_model()

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_evaluation():

    if os.path.exists(EVALUATION_PATH):
        return joblib.load(EVALUATION_PATH)

    return None


try:
    model = load_model()
    evaluation = load_evaluation()

except Exception as error:
    st.error(f"Unable to load the AI model: {error}")
    st.stop()


# ----------------------------------
# SESSION STATE
# ----------------------------------

if "history" not in st.session_state:
    st.session_state.history = []


# ----------------------------------
# HEADER
# ----------------------------------

st.markdown(
    '<div class="main-title">📩 AI SMS SPAM DETECTION</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Message Classification using NLP and Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ----------------------------------
# SIDEBAR
# ----------------------------------

st.sidebar.title("🛡️ SMS Security")

page = st.sidebar.radio(
    "Navigation",
    [
        "Message Detection",
        "Model Performance",
        "Prediction History",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.success("AI Model Loaded Successfully")

st.sidebar.caption(
    "Algorithm: Logistic Regression"
)

st.sidebar.caption(
    "Feature Extraction: TF-IDF"
)


# ----------------------------------
# PAGE 1: MESSAGE DETECTION
# ----------------------------------

if page == "Message Detection":

    st.header("🔍 SMS Message Analyzer")

    st.write(
        "Enter an SMS message to identify whether it is "
        "Spam or Legitimate."
    )

    message = st.text_area(
        "Enter your SMS message",
        placeholder="Example: Congratulations! You won a free prize...",
        height=140
    )

    if st.button("🚀 Analyze Message", use_container_width=True):

        if not message.strip():

            st.warning("Please enter a message.")

        else:

            prediction = model.predict([message])[0]

            probabilities = model.predict_proba([message])[0]

            classes = list(model.classes_)

            spam_probability = probabilities[classes.index(1)] * 100

            ham_probability = probabilities[classes.index(0)] * 100

            if prediction == 1:

                st.markdown(
                    """
                    <div class="result-box spam">
                        <h2>🚨 SPAM MESSAGE DETECTED</h2>
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

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Spam Probability",
                    f"{spam_probability:.2f}%"
                )

            with col2:
                st.metric(
                    "Legitimate Probability",
                    f"{ham_probability:.2f}%"
                )

            st.subheader("Prediction Analysis")

            st.progress(float(spam_probability / 100))

            st.caption("Spam probability")

            st.progress(float(ham_probability / 100))

            st.caption("Legitimate probability")

            st.session_state.history.insert(
                0,
                {
                    "Message": message,
                    "Prediction": (
                        "Spam" if prediction == 1
                        else "Legitimate"
                    ),
                    "Spam Probability": round(
                        spam_probability, 2
                    )
                }
            )


    st.divider()

    st.subheader("💡 Try Sample Messages")

    examples = {
        "Spam Example": (
            "Congratulations! You have won a free prize. "
            "Call now to claim your reward!"
        ),

        "Legitimate Example": (
            "Hey, are we meeting today at 5 PM?"
        )
    }

    for name, example in examples.items():

        st.write(f"**{name}**")

        st.code(example)


# ----------------------------------
# PAGE 2: MODEL PERFORMANCE
# ----------------------------------

elif page == "Model Performance":

    st.header("📊 Model Performance Dashboard")

    if evaluation:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Accuracy",
            f"{evaluation['accuracy'] * 100:.2f}%"
        )

        col2.metric(
            "Precision",
            f"{evaluation['precision'] * 100:.2f}%"
        )

        col3.metric(
            "Recall",
            f"{evaluation['recall'] * 100:.2f}%"
        )

        col4.metric(
            "F1-Score",
            f"{evaluation['f1_score'] * 100:.2f}%"
        )

        st.divider()

        st.subheader("Confusion Matrix")

        cm = evaluation["confusion_matrix"]

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Legitimate", "Spam"],
            yticklabels=["Legitimate", "Spam"],
            ax=ax
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")

        st.pyplot(fig)
        plt.close(fig)

        col1, col2 = st.columns(2)

        col1.metric(
            "Training Samples",
            evaluation["training_size"]
        )

        col2.metric(
            "Testing Samples",
            evaluation["testing_size"]
        )

    else:

        st.warning("Model evaluation data is unavailable.")


# ----------------------------------
# PAGE 3: PREDICTION HISTORY
# ----------------------------------

elif page == "Prediction History":

    st.header("📜 Prediction History")

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )

        if st.button("Clear History"):

            st.session_state.history = []

            st.rerun()

    else:

        st.info(
            "No predictions yet. Analyze some SMS messages first."
        )


# ----------------------------------
# PAGE 4: ABOUT PROJECT
# ----------------------------------

elif page == "About Project":

    st.header("ℹ️ About This Project")

    st.markdown("""
    ### AI-Powered SMS Spam Detection

    This project uses Natural Language Processing and
    Machine Learning to classify SMS messages.

    **Technology Stack**

    - Python
    - Natural Language Processing
    - Scikit-learn
    - TF-IDF Vectorization
    - Logistic Regression
    - Streamlit
    - Pandas
    - Matplotlib
    - Seaborn

    **Model Workflow**

    1. SMS Dataset Collection
    2. Text Preprocessing
    3. TF-IDF Feature Extraction
    4. Logistic Regression Training
    5. Model Evaluation
    6. SMS Classification

    **Classification Categories**

    - Spam
    - Legitimate (Ham)
    """)

    st.info(
        "This application is an educational machine learning "
        "prototype. Predictions are not guaranteed to be correct."
    )


# ----------------------------------
# FOOTER
# ----------------------------------

st.divider()

st.caption(
    "AI-Powered SMS Spam Detection | Python | NLP | Scikit-learn"
)
