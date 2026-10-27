"""
================================================================================
BAMAT207: Probability and Statistics - Course Project
Topic: Probability and Random Variables
Project Title: Probabilistic SMS Spam Detection
Institution: Vellore Institute of Technology (VIT)
================================================================================
"""

import os
import re
import urllib.request
import zipfile
import io
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Ensure output directory for figures
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_dataset():
    """Load or download the SMS Spam Collection dataset."""
    dataset_path = os.path.join(OUTPUT_DIR, "SMSSpamCollection")
    if not os.path.exists(dataset_path):
        print("[*] Dataset not found locally. Downloading from UCI Machine Learning Repository...")
        uci_url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
        try:
            req = urllib.request.Request(uci_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                z = zipfile.ZipFile(io.BytesIO(resp.read()))
                z.extractall(OUTPUT_DIR)
            print("[+] Successfully downloaded and extracted dataset from UCI.")
        except Exception as e:
            print(f"[!] UCI direct download failed ({e}). Falling back to GitHub mirror...")
            mirror_url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
            req = urllib.request.Request(mirror_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                with open(dataset_path, "wb") as f:
                    f.write(resp.read())
            print("[+] Successfully downloaded dataset from mirror.")
    
    # Load into DataFrame
    df = pd.read_csv(dataset_path, sep="\t", names=["label", "message"], header=None, encoding="utf-8")
    return df


def main():
    print("=" * 70)
    print("      BAMAT207: PROBABILISTIC SMS SPAM DETECTION SYSTEM")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Dataset Handling
    # ---------------------------------------------------------
    print("\n[ACTIVITY 1] Dataset Handling & Preprocessing")
    df = load_dataset()
    total_raw = len(df)
    missing_vals = df.isnull().sum().to_dict()
    duplicates_count = int(df.duplicated().sum())

    print(f"  - Initial Total Records: {total_raw}")
    print(f"  - Missing Values: {missing_vals}")
    print(f"  - Duplicate Records Found: {duplicates_count}")

    # Remove duplicates
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    total_cleaned = len(df_cleaned)
    print(f"  - Cleaned Records Count: {total_cleaned}")

    # Convert labels into numerical values: Spam = 1, Ham = 0
    df_cleaned["label_num"] = df_cleaned["label"].map({"ham": 0, "spam": 1})
    print("  - Encoded Labels: Ham -> 0, Spam -> 1")

    # ---------------------------------------------------------
    # 2. Dataset Analysis
    # ---------------------------------------------------------
    print("\n[ACTIVITY 2] Dataset Analysis & Class Probabilities")
    n_spam = int((df_cleaned["label_num"] == 1).sum())
    n_ham = int((df_cleaned["label_num"] == 0).sum())
    total_msgs = len(df_cleaned)

    p_spam = n_spam / total_msgs
    p_ham = n_ham / total_msgs

    print(f"  - Genuine (Ham) Messages: {n_ham} ({p_ham * 100:.2f}%)")
    print(f"  - Spam Messages:          {n_spam} ({p_spam * 100:.2f}%)")
    print(f"  - P(Spam) = {n_spam}/{total_msgs} = {p_spam:.4f}")
    print(f"  - P(Ham)  = {n_ham}/{total_msgs} = {p_ham:.4f}")

    # Class distribution chart
    plt.figure(figsize=(7, 5))
    bars = plt.bar(["Ham (Genuine)", "Spam"], [n_ham, n_spam], color=["#2563eb", "#dc2626"], width=0.45, edgecolor="black", alpha=0.85)
    plt.title("Class Distribution of Cleaned SMS Dataset", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Number of Messages", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars:
        yval = bar.get_height()
        pct = (yval / total_msgs) * 100
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f"{yval}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.ylim(0, max(n_ham, n_spam) * 1.15)
    plt.tight_layout()
    dist_plot_path = os.path.join(PLOTS_DIR, "class_distribution.png")
    plt.savefig(dist_plot_path, dpi=300)
    plt.close()
    print(f"  - Class distribution chart saved at: {dist_plot_path}")

    # ---------------------------------------------------------
    # 3. Word-Based Probability Analysis
    # ---------------------------------------------------------
    print("\n[ACTIVITY 3] Word-Based Probability Analysis & Bayes' Theorem Verification")
    suspicious_words = ["free", "prize", "win", "urgent", "offer"]
    word_results = []

    print(f"{'Word':<10} | {'Total':<6} | {'Spam':<5} | {'Ham':<5} | {'P(Word)':<9} | {'P(Word|Spam)':<12} | {'P(Spam|Word) Emp':<16} | {'P(Spam|Word) Bayes':<18} | {'Match'}")
    print("-" * 102)

    for word in suspicious_words:
        pattern = r"\b" + re.escape(word) + r"\b"
        contains_word = df_cleaned["message"].str.contains(pattern, case=False, regex=True)

        n_word_total = int(contains_word.sum())
        n_word_spam = int((contains_word & (df_cleaned["label_num"] == 1)).sum())
        n_word_ham = int((contains_word & (df_cleaned["label_num"] == 0)).sum())

        p_word = n_word_total / total_msgs
        p_word_given_spam = n_word_spam / n_spam if n_spam > 0 else 0.0
        p_word_given_ham = n_word_ham / n_ham if n_ham > 0 else 0.0

        # Direct Empirical Conditional Probability
        p_spam_given_word_emp = n_word_spam / n_word_total if n_word_total > 0 else 0.0

        # Bayes' Theorem:
        # P(Spam | Word) = [P(Word | Spam) * P(Spam)] / P(Word)
        p_spam_given_word_bayes = (p_word_given_spam * p_spam) / p_word if p_word > 0 else 0.0

        is_match = np.isclose(p_spam_given_word_emp, p_spam_given_word_bayes)

        print(f"{word:<10} | {n_word_total:<6} | {n_word_spam:<5} | {n_word_ham:<5} | {p_word:<9.4f} | {p_word_given_spam:<12.4f} | {p_spam_given_word_emp:<16.4f} | {p_spam_given_word_bayes:<18.4f} | {'YES' if is_match else 'NO'}")

        word_results.append({
            "word": word,
            "total_occurrences": n_word_total,
            "spam_occurrences": n_word_spam,
            "ham_occurrences": n_word_ham,
            "p_word": p_word,
            "p_word_given_spam": p_word_given_spam,
            "p_word_given_ham": p_word_given_ham,
            "p_spam_given_word_empirical": p_spam_given_word_emp,
            "p_spam_given_word_bayes": p_spam_given_word_bayes
        })

    # Plot word probabilities comparison
    words_list = [r["word"].upper() for r in word_results]
    p_spam_vals = [r["p_spam_given_word_empirical"] for r in word_results]
    p_ham_vals = [1.0 - v for v in p_spam_vals]

    x = np.arange(len(words_list))
    width = 0.35

    plt.figure(figsize=(8.5, 5))
    plt.bar(x - width/2, p_spam_vals, width, label="P(Spam | Word)", color="#dc2626", alpha=0.85, edgecolor="black")
    plt.bar(x + width/2, p_ham_vals, width, label="P(Ham | Word)", color="#2563eb", alpha=0.85, edgecolor="black")
    plt.xlabel("Suspicious Word", fontsize=11, fontweight="bold")
    plt.ylabel("Probability", fontsize=11, fontweight="bold")
    plt.title("Posterior Probabilities: P(Spam | Word) vs P(Ham | Word)", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(x, words_list, fontweight="bold")
    plt.ylim(0, 1.15)
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    for i in range(len(words_list)):
        plt.text(x[i] - width/2, p_spam_vals[i] + 0.02, f"{p_spam_vals[i]:.2f}", ha="center", fontsize=9, fontweight="bold")
        plt.text(x[i] + width/2, p_ham_vals[i] + 0.02, f"{p_ham_vals[i]:.2f}", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    word_plot_path = os.path.join(PLOTS_DIR, "word_probabilities.png")
    plt.savefig(word_plot_path, dpi=300)
    plt.close()
    print(f"  - Word probability chart saved at: {word_plot_path}")

    # ---------------------------------------------------------
    # 4. Random Variable Analysis
    # ---------------------------------------------------------
    print("\n[ACTIVITY 4] Random Variable Analysis (Bernoulli Distribution)")
    p = p_spam
    expected_value_Y = p
    variance_Y = p * (1 - p)
    std_dev_Y = np.sqrt(variance_Y)

    print(f"  - Discrete Random Variable Y:")
    print(f"      Y = 1 if incoming message is Spam")
    print(f"      Y = 0 if incoming message is Ham (Genuine)")
    print(f"  - Parameter p = P(Spam) = {p:.4f}")
    print(f"  - Expectation E(Y)      = p = {expected_value_Y:.4f}")
    print(f"  - Variance Var(Y)       = p * (1 - p) = {p:.4f} * (1 - {p:.4f}) = {variance_Y:.4f}")
    print(f"  - Standard Dev SD(Y)    = sqrt(Var(Y)) = {std_dev_Y:.4f}")

    # ---------------------------------------------------------
    # 5. Data Preparation
    # ---------------------------------------------------------
    print("\n[ACTIVITY 5] Data Preparation & Vectorization")
    X = df_cleaned["message"]
    y = df_cleaned["label_num"]

    # 80% train, 20% test with stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  - Total Samples: {len(X)}")
    print(f"  - Training Set (80%): {len(X_train)} samples")
    print(f"      Train Ham: {(y_train == 0).sum()}, Train Spam: {(y_train == 1).sum()} (Spam Ratio: {(y_train == 1).mean():.4f})")
    print(f"  - Testing Set (20%):  {len(X_test)} samples")
    print(f"      Test Ham:  {(y_test == 0).sum()}, Test Spam:  {(y_test == 1).sum()} (Spam Ratio: {(y_test == 1).mean():.4f})")

    # Feature extraction with CountVectorizer
    vectorizer = CountVectorizer(stop_words="english", lowercase=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"  - Vocabulary Size extracted: {X_train_vec.shape[1]} unique word features")

    # ---------------------------------------------------------
    # 6. Model Building
    # ---------------------------------------------------------
    print("\n[ACTIVITY 6] Model Building: Multinomial Naive Bayes")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    print("  - MultinomialNB model successfully trained on word-count feature matrix.")

    # ---------------------------------------------------------
    # 7. Model Evaluation
    # ---------------------------------------------------------
    print("\n[ACTIVITY 7] Model Evaluation & Metrics")
    y_pred = model.predict(X_test_vec)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"  - Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  - Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"  - Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"  - F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print("\n  - Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

    print("  - Confusion Matrix Breakdown:")
    print(f"      * True Negatives (TN):  {tn} (Correctly classified genuine messages)")
    print(f"      * False Positives (FP): {fp} (Genuine messages incorrectly classified as spam)")
    print(f"      * False Negatives (FN): {fn} (Spam messages incorrectly classified as genuine)")
    print(f"      * True Positives (TP):  {tp} (Correctly classified spam messages)")

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Multinomial Naive Bayes)", fontsize=12, fontweight="bold", pad=12)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Predicted Ham", "Predicted Spam"], fontsize=10)
    plt.yticks(tick_marks, ["Actual Ham", "Actual Spam"], fontsize=10)

    # Label values in matrix
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            label_text = f"{val}\n"
            if i == 0 and j == 0:
                label_text += "(TN)"
            elif i == 0 and j == 1:
                label_text += "(FP)"
            elif i == 1 and j == 0:
                label_text += "(FN)"
            else:
                label_text += "(TP)"
            plt.text(j, i, label_text, horizontalalignment="center", verticalalignment="center",
                     color="white" if val > thresh else "black", fontsize=11, fontweight="bold")
    plt.ylabel("Actual Label", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_plot_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"  - Confusion matrix chart saved at: {cm_plot_path}")

    # ---------------------------------------------------------
    # 8. Real-Time Testing
    # ---------------------------------------------------------
    print("\n[ACTIVITY 8] Real-Time Testing & Inference")
    sample_messages = [
        "Congratulations! You have won a free prize. Claim now.",
        "Hey bro, are we meeting at the campus library this evening?",
        "URGENT! Your mobile number won 1000 prize. Call now to claim offer.",
        "Can you send me the lecture notes for probability and statistics?"
    ]

    realtime_results = []
    for msg in sample_messages:
        vec = vectorizer.transform([msg])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        lbl = "Spam" if pred == 1 else "Ham (Genuine)"
        print(f"\n  Incoming SMS: \"{msg}\"")
        print(f"  Prediction:   {lbl}")
        print(f"  P(Spam | Message) = {prob[1]:.4f}")
        print(f"  P(Ham | Message)  = {prob[0]:.4f}")
        realtime_results.append({
            "message": msg,
            "prediction": lbl,
            "p_spam": float(prob[1]),
            "p_ham": float(prob[0])
        })

    # Save summary data for report generator
    summary_data = {
        "dataset": {
            "total_raw": total_raw,
            "duplicates": int(duplicates_count),
            "total_cleaned": total_cleaned,
            "n_ham": n_ham,
            "n_spam": n_spam,
            "p_ham": float(p_ham),
            "p_spam": float(p_spam)
        },
        "words": word_results,
        "random_variable": {
            "p": float(p),
            "E_Y": float(expected_value_Y),
            "Var_Y": float(variance_Y),
            "Std_Y": float(std_dev_Y)
        },
        "model_evaluation": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "realtime": realtime_results
    }

    results_file = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_file, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"\n[+] Execution completed successfully! Results recorded in {results_file}")


if __name__ == "__main__":
    main()
