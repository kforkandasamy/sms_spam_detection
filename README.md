# Probabilistic SMS Spam Detection System
**Course:** BAMAT207 - Probability and Statistics  
**Topic:** Probability and Random Variables  
**Institution:** Vellore Institute of Technology (VIT)  

---

## 📌 Project Overview
This project develops an end-to-end Python system that classifies SMS text messages as either **Spam** or **Genuine (Ham)**. It implements foundational principles of:
- **Prior & Conditional Probability**
- **Bayes' Theorem Verification** for suspicious keywords
- **Bernoulli Random Variable Modeling** (Expectation & Variance)
- **Multinomial Naive Bayes Machine Learning** with text tokenization
- **Real-Time Classification & Posterior Inference**

---

## 📂 Project Structure
```
sms-spam-detection/
│── SMSSpamCollection          # UCI dataset (5,574 labelled messages)
│── sms_spam_detection.py      # Standalone Python script running all 8 activities
│── sms_spam_detection.ipynb   # Jupyter / Google Colab notebook
│── PROJECT_REPORT.md          # Comprehensive academic project report
│── PROJECT_REPORT.html        # Formatted report ready to view or print to PDF
│── results.json               # Extracted metrics and analytical calculations
│── plots/
│   │── class_distribution.png # Bar chart of Ham vs Spam distribution
│   │── word_probabilities.png # Posterior probabilities P(Spam|Word) vs P(Ham|Word)
│   │── confusion_matrix.png   # Model confusion matrix (TN, FP, FN, TP)
│   └── vit_logo.png           # VIT institution header emblem
└── README.md                  # Project documentation and setup guide
```

---

## 🚀 How to Run

### Option 1: Running the Python Script
Ensure Python 3.9+ is installed along with required packages:
```bash
pip install pandas numpy matplotlib scikit-learn
```

Run the pipeline:
```bash
python sms_spam_detection.py
```

### Option 2: Running in Google Colab or Jupyter Notebook
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `sms_spam_detection.ipynb`.
3. Select **Runtime > Run all**.

---

## 📊 Summary of Results
| Metric / Calculation | Value |
| :--- | :--- |
| **Initial Records** | 5,572 |
| **Cleaned Records (Duplicates Removed)** | 5,169 |
| **P(Ham)** | 0.8737 (87.37%) |
| **P(Spam)** | 0.1263 (12.63%) |
| **Random Variable $E[Y]$** | 0.1263 |
| **Random Variable $\text{Var}(Y)$** | 0.1104 |
| **Model Test Accuracy** | **98.36%** |
| **Model Precision** | **95.97%** |
| **Model Recall** | **90.84%** |
| **Model F1-Score** | **93.33%** |
| **False Positives (Ham classified as Spam)** | 5 |
| **False Negatives (Spam classified as Ham)** | 12 |

---

## 📜 Deliverables
- [x] Python script (`sms_spam_detection.py`)
- [x] Google Colab / Jupyter notebook (`sms_spam_detection.ipynb`)
- [x] Cleaned dataset summary
- [x] Class distribution chart
- [x] Bayes' theorem verification for suspicious words (`free`, `prize`, `win`, `urgent`, `offer`)
- [x] Random variable expectation & variance calculations
- [x] Trained Multinomial Naive Bayes model & Confusion Matrix
- [x] Real-time inference testing
- [x] Final Academic Project Report (`PROJECT_REPORT.md` & `PROJECT_REPORT.html`)
