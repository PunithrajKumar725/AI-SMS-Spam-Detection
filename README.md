# 📩 AI-Powered SMS Spam Detection

**Python | Natural Language Processing | Scikit-learn | TF-IDF | Logistic Regression | Streamlit**

## 📌 Project Overview

AI-Powered SMS Spam Detection is a machine learning application that automatically classifies SMS messages as **Spam** or **Legitimate (Ham)** using Natural Language Processing (NLP).

The project uses TF-IDF vectorization to convert text messages into numerical features and Logistic Regression to predict whether a message is spam or legitimate.

An interactive Streamlit web application allows users to enter SMS messages and instantly receive classification results.

## 🎯 Objectives

- Detect unwanted and fraudulent SMS messages.
- Apply NLP techniques for text preprocessing and feature extraction.
- Build a machine learning classification model.
- Evaluate model performance using standard classification metrics.
- Develop an interactive web application for real-time message classification.

## 🚀 Key Features

- SMS message classification.
- Text feature extraction using TF-IDF.
- Logistic Regression machine learning algorithm.
- Spam and legitimate message prediction.
- Interactive Streamlit user interface.
- Model evaluation using precision, recall and F1-score.
- Confusion matrix visualization.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| NLP | Text processing |
| Scikit-learn | Machine learning |
| TF-IDF | Text feature extraction |
| Logistic Regression | Classification algorithm |
| Streamlit | Interactive web application |
| Pandas | Dataset processing |
| Matplotlib / Seaborn | Model evaluation visualization |

## ⚙️ System Workflow

1. **Dataset Collection:** Load the SMS dataset containing spam and legitimate messages.
2. **Text Preprocessing:** Clean and prepare the SMS messages.
3. **Feature Extraction:** Convert text into numerical features using TF-IDF.
4. **Model Training:** Train the Logistic Regression classification model.
5. **Model Evaluation:** Evaluate performance using precision, recall, F1-score and confusion matrix.
6. **Prediction:** Classify new SMS messages.
7. **Web Application:** Display predictions through the Streamlit interface.

## 📊 Model Evaluation

The classification model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

These metrics help measure the model's ability to distinguish spam messages from legitimate messages.

## 💻 Installation and Execution

### Prerequisites

- Python 3.10 or later
- VS Code (recommended)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/AI-SMS-Spam-Detection.git
```

### Step 2: Navigate to the Project Folder

```bash
cd AI-SMS-Spam-Detection
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Environment

Windows:

```bash
venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🔮 Future Enhancements

- Advanced NLP preprocessing.
- Comparison of multiple machine learning algorithms.
- Improved detection of phishing and fraudulent messages.
- Confidence score visualization.
- Deployment through Streamlit Community Cloud.

## 👨‍💻 Project Category

Artificial Intelligence | Machine Learning | Natural Language Processing | Text Classification

## 📜 License

This project is available under the MIT License.

---

**Built with Python, Machine Learning and NLP to identify unwanted SMS messages and improve digital communication safety.**
