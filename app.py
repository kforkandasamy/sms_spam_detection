import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(
    page_title="SMS Spam Detector - BAMAT207",
    page_icon="📱",
    layout="wide"
)

# Load data and results
with open("results.json", "r") as f:
    results = json.load(f)

@st.cache_resource
def get_trained_model():
    df = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "message"], header=None)
    df = df.drop_duplicates().reset_index(drop=True)
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    X = df["message"]
    y = df["label_num"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    vec = CountVectorizer(stop_words="english", lowercase=True)
    X_train_vec = vec.fit_transform(X_train)
    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)
    return vec, clf

vectorizer, model = get_trained_model()

# Header
st.title("📱 Probabilistic SMS Spam Detection System")
st.markdown("**BAMAT207: Probability & Statistics Course Project** | **Vellore Institute of Technology (VIT)**")
st.markdown("Student: **Sai Shreya** | Registration No.: **25BCE2074**")
st.divider()

# Sidebar: Quick Info & PDF Download
with st.sidebar:
    st.image("plots/vit_logo.png", use_container_width=True)
    st.header("Project Details")
    st.write("**Topic:** Probability & Random Variables")
    st.write("**Dataset:** UCI SMS Spam Collection (5,169 cleaned)")
    st.write("**Model:** Multinomial Naive Bayes")
    st.write("**Test Accuracy:** 98.36%")
    st.write("**Precision:** 95.97%")
    
    st.divider()
    pdf_path = "BAMAT207_SMS_Spam_Detection_Project_Report.pdf"
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📄 Download Official PDF Report",
                data=f,
                file_name="BAMAT207_SMS_Spam_Detection_Project_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

tab1, tab2, tab3 = st.tabs(["🚀 Live SMS Classifier", "📊 Probability & Statistics Dashboard", "📜 Confusion Matrix & Metrics"])

with tab1:
    st.subheader("Test Incoming SMS in Real-Time")
    st.write("Enter an SMS message below or pick one of the sample messages to inspect prediction probabilities:")
    
    presets = [
        "Congratulations! You have won a free prize. Claim now.",
        "Hey bro, are we meeting at the campus library this evening?",
        "URGENT! Your mobile number won 1000 prize. Call now to claim offer.",
        "Can you send me the lecture notes for probability and statistics?",
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005."
    ]
    
    selected_preset = st.selectbox("Select a sample message:", ["-- Choose a preset or type below --"] + presets)
    
    default_text = "" if selected_preset.startswith("--") else selected_preset
    user_input = st.text_area("SMS Message Content:", value=default_text, placeholder="Type or paste SMS message here...")
    
    if st.button("Classify SMS", type="primary"):
        if user_input.strip():
            vec = vectorizer.transform([user_input])
            pred = model.predict(vec)[0]
            probs = model.predict_proba(vec)[0]
            
            p_ham = float(probs[0])
            p_spam = float(probs[1])
            
            col1, col2 = st.columns(2)
            with col1:
                if pred == 1:
                    st.error(f"🚨 **Prediction: SPAM**")
                else:
                    st.success(f"✅ **Prediction: GENUINE (HAM)**")
            
            with col2:
                st.metric("P(Spam | Message)", f"{p_spam * 100:.2f}%")
                st.metric("P(Ham | Message)", f"{p_ham * 100:.2f}%")
            
            st.progress(p_spam, text=f"Spam Probability: {p_spam * 100:.2f}%")
        else:
            st.warning("Please enter some text to classify.")

with tab2:
    st.subheader("Probability & Random Variable Analysis")
    colA, colB, colC = st.columns(3)
    colA.metric("P(Spam) Prior", f"{results['dataset']['p_spam']:.4f} (12.63%)")
    colB.metric("Random Variable E[Y]", f"{results['random_variable']['E_Y']:.4f}")
    colC.metric("Random Variable Var(Y)", f"{results['random_variable']['Var_Y']:.4f}")
    
    st.divider()
    st.write("### Bayes' Theorem Word Verification")
    st.latex(r"P(\text{Spam} \mid \text{Word}) = \frac{P(\text{Word} \mid \text{Spam}) \cdot P(\text{Spam})}{P(\text{Word})}")
    
    w_df = pd.DataFrame(results["words"])
    w_df = w_df[["word", "total_occurrences", "spam_occurrences", "ham_occurrences", "p_word", "p_word_given_spam", "p_spam_given_word_empirical", "p_spam_given_word_bayes"]]
    w_df.columns = ["Keyword", "Total Count", "Spam Count", "Ham Count", "P(Word)", "P(Word|Spam)", "Empirical P(Spam|Word)", "Bayes P(Spam|Word)"]
    st.dataframe(w_df, use_container_width=True)
    
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image("plots/class_distribution.png", caption="Class Distribution", use_container_width=True)
    with col_img2:
        st.image("plots/word_probabilities.png", caption="Word Posterior Probabilities", use_container_width=True)

with tab3:
    st.subheader("Model Performance & Confusion Matrix")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "98.36%")
    col2.metric("Precision", "95.97%")
    col3.metric("Recall", "90.84%")
    col4.metric("F1-Score", "93.33%")
    
    st.divider()
    col_cm, col_desc = st.columns([1, 1])
    with col_cm:
        st.image("plots/confusion_matrix.png", caption="Confusion Matrix Heatmap", use_container_width=True)
    with col_desc:
        st.write("### Test Evaluation Breakdown (1,034 Test Samples):")
        st.write(f"- **True Negatives (TN):** {results['model_evaluation']['tn']} (Genuine correctly classified)")
        st.write(f"- **False Positives (FP):** {results['model_evaluation']['fp']} (Genuine incorrectly flagged as spam)")
        st.write(f"- **False Negatives (FN):** {results['model_evaluation']['fn']} (Spam incorrectly marked genuine)")
        st.write(f"- **True Positives (TP):** {results['model_evaluation']['tp']} (Spam correctly caught)")
        st.info("💡 **Why Precision Matters:** With only 5 False Positives out of 903 genuine test messages (0.55% false alarm rate), the system ensures critical interpersonal messages are preserved.")
