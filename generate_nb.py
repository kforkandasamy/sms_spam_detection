import json

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# BAMAT207: Probability and Statistics - Course Project\n",
    "## Project Title: Probabilistic SMS Spam Detection\n",
    "**Topic:** Probability and Random Variables  \n",
    "**Institution:** Vellore Institute of Technology (VIT)  \n",
    "**Dataset:** UCI Machine Learning Repository - SMS Spam Collection (5,574 messages)\n",
    "\n",
    "---\n",
    "### Project Objective:\n",
    "Develop a Python-based system that classifies incoming SMS messages as **Spam** or **Genuine (Ham)** by analyzing word distributions, calculating prior and posterior probabilities using **Bayes' Theorem**, modeling message classification with **Bernoulli Random Variables**, and training a **Multinomial Naive Bayes** classifier."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 0: Import Required Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import re\n",
    "import urllib.request\n",
    "import zipfile\n",
    "import io\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report\n",
    "\n",
    "print('All libraries successfully imported!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Activity 1: Dataset Handling\n",
    "- Load SMS messages and labels from UCI\n",
    "- Check missing values and remove duplicate messages\n",
    "- Encode labels: `Spam = 1`, `Ham = 0`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Download dataset if not present\n",
    "dataset_url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'\n",
    "df = pd.read_csv(dataset_url, sep='\\t', names=['label', 'message'], header=None)\n",
    "\n",
    "print(f'Initial total records: {len(df)}')\n",
    "print('Missing values:\\n', df.isnull().sum())\n",
    "duplicates = df.duplicated().sum()\n",
    "print(f'Duplicates count: {duplicates}')\n",
    "\n",
    "# Clean dataset\n",
    "df_cleaned = df.drop_duplicates().reset_index(drop=True)\n",
    "print(f'Cleaned dataset records: {len(df_cleaned)}')\n",
    "\n",
    "# Numerical encoding\n",
    "df_cleaned['label_num'] = df_cleaned['label'].map({'ham': 0, 'spam': 1})\n",
    "df_cleaned.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Activity 2: Dataset Analysis\n",
    "- Compute marginal prior probabilities $P(\\text{Spam})$ and $P(\\text{Ham})$\n",
    "- Visualize class distribution with a bar chart"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "n_spam = int((df_cleaned['label_num'] == 1).sum())\n",
    "n_ham = int((df_cleaned['label_num'] == 0).sum())\n",
    "total_msgs = len(df_cleaned)\n",
    "\n",
    "p_spam = n_spam / total_msgs\n",
    "p_ham = n_ham / total_msgs\n",
    "\n",
    "print(f'Genuine (Ham) Messages: {n_ham} ({p_ham*100:.2f}%)')\n",
    "print(f'Spam Messages:          {n_spam} ({p_spam*100:.2f}%)')\n",
    "print(f'P(Spam) = {p_spam:.4f}')\n",
    "print(f'P(Ham)  = {p_ham:.4f}')\n",
    "\n",
    "# Plot Class Distribution\n",
    "plt.figure(figsize=(7, 5))\n",
    "bars = plt.bar(['Ham (Genuine)', 'Spam'], [n_ham, n_spam], color=['#2563eb', '#dc2626'], width=0.45, edgecolor='black', alpha=0.85)\n",
    "plt.title('Class Distribution of Cleaned SMS Dataset', fontsize=13, fontweight='bold', pad=12)\n",
    "plt.ylabel('Number of Messages', fontsize=11)\n",
    "plt.grid(axis='y', linestyle='--', alpha=0.4)\n",
    "for bar in bars:\n",
    "    yval = bar.get_height()\n",
    "    pct = (yval / total_msgs) * 100\n",
    "    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f'{yval}\\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')\n",
    "plt.ylim(0, max(n_ham, n_spam) * 1.15)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Activity 3: Word-Based Probability Analysis\n",
    "- Analyze 5 suspicious keywords: `free`, `prize`, `win`, `urgent`, `offer`\n",
    "- Compute empirical conditional probability $P(\\text{Spam} \\mid \\text{Word})$\n",
    "- Verify results analytically using Bayes' Theorem:\n",
    "$$P(\\text{Spam} \\mid \\text{Word}) = \\frac{P(\\text{Word} \\mid \\text{Spam}) \\cdot P(\\text{Spam})}{P(\\text{Word})}$$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "suspicious_words = ['free', 'prize', 'win', 'urgent', 'offer']\n",
    "word_records = []\n",
    "\n",
    "print(f\"{'Word':<10} | {'Total':<6} | {'Spam':<5} | {'Ham':<5} | {'P(Word)':<9} | {'P(W|Spam)':<10} | {'Empirical P':<12} | {'Bayes P':<10} | {'Match'}\")\n",
    "print('-' * 88)\n",
    "\n",
    "for word in suspicious_words:\n",
    "    pattern = r'\\b' + re.escape(word) + r'\\b'\n",
    "    contains_word = df_cleaned['message'].str.contains(pattern, case=False, regex=True)\n",
    "    n_word_total = int(contains_word.sum())\n",
    "    n_word_spam = int((contains_word & (df_cleaned['label_num'] == 1)).sum())\n",
    "    n_word_ham = int((contains_word & (df_cleaned['label_num'] == 0)).sum())\n",
    "    \n",
    "    p_word = n_word_total / total_msgs\n",
    "    p_word_given_spam = n_word_spam / n_spam if n_spam > 0 else 0.0\n",
    "    \n",
    "    p_spam_given_word_emp = n_word_spam / n_word_total if n_word_total > 0 else 0.0\n",
    "    p_spam_given_word_bayes = (p_word_given_spam * p_spam) / p_word if p_word > 0 else 0.0\n",
    "    \n",
    "    match = np.isclose(p_spam_given_word_emp, p_spam_given_word_bayes)\n",
    "    print(f\"{word:<10} | {n_word_total:<6} | {n_word_spam:<5} | {n_word_ham:<5} | {p_word:<9.4f} | {p_word_given_spam:<10.4f} | {p_spam_given_word_emp:<12.4f} | {p_spam_given_word_bayes:<10.4f} | {'YES' if match else 'NO'}\")\n",
    "    word_records.append({'Word': word.upper(), 'P(Spam|Word)': p_spam_given_word_emp, 'P(Ham|Word)': 1 - p_spam_given_word_emp})\n",
    "\n",
    "# Plot word posterior probabilities\n",
    "w_df = pd.DataFrame(word_records)\n",
    "x = np.arange(len(w_df))\n",
    "plt.figure(figsize=(8, 4.5))\n",
    "plt.bar(x - 0.175, w_df['P(Spam|Word)'], 0.35, label='P(Spam | Word)', color='#dc2626', alpha=0.85, edgecolor='black')\n",
    "plt.bar(x + 0.175, w_df['P(Ham|Word)'], 0.35, label='P(Ham | Word)', color='#2563eb', alpha=0.85, edgecolor='black')\n",
    "plt.xticks(x, w_df['Word'], fontweight='bold')\n",
    "plt.ylabel('Posterior Probability', fontweight='bold')\n",
    "plt.title('Posterior Probability of Spam vs Ham Given Suspicious Words', fontweight='bold', pad=12)\n",
    "plt.ylim(0, 1.15)\n",
    "plt.legend()\n",
    "plt.grid(axis='y', linestyle='--', alpha=0.4)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Activity 4: Random Variable Analysis\n",
    "Let $Y$ be a discrete random variable indicating whether an incoming message is spam:\n",
    "$$Y = \\begin{cases} 1 & \\text{if message is Spam} \\\\ 0 & \\text{if message is Ham (Genuine)} \\end{cases}$$\n",
    "\n",
    "Since $Y \\sim \\text{Bernoulli}(p)$ where $p = P(\\text{Spam})$:\n",
    "- **Expectation:** $E[Y] = p$\n",
    "- **Variance:** $\\text{Var}(Y) = p(1-p)$\n",
    "- **Standard Deviation:** $\\sigma_Y = \\sqrt{p(1-p)}$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "p = p_spam\n",
    "E_Y = p\n",
    "Var_Y = p * (1 - p)\n",
    "Std_Y = np.sqrt(Var_Y)\n",
    "\n",
    "print('Random Variable Y ~ Bernoulli(p):')\n",
    "print(f'  Parameter p  = {p:.4f}')\n",
    "print(f'  E[Y]         = {E_Y:.4f}')\n",
    "print(f'  Var(Y)       = {Var_Y:.4f}')\n",
    "print(f'  Std Dev(Y)   = {Std_Y:.4f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Activity 5 & 6: Data Preparation & Model Building\n",
    "- Split dataset: 80% train, 20% test using stratified sampling (`stratify=y`)\n",
    "- Extract discrete word-frequency features using `CountVectorizer`\n",
    "- Train `MultinomialNB` classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df_cleaned['message']\n",
    "y = df_cleaned['label_num']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)\n",
    "\n",
    "vectorizer = CountVectorizer(stop_words='english', lowercase=True)\n",
    "X_train_vec = vectorizer.fit_transform(X_train)\n",
    "X_test_vec = vectorizer.transform(X_test)\n",
    "\n",
    "print(f'Training samples: {len(X_train)}, Testing samples: {len(X_test)}')\n",
    "print(f'Vocabulary features: {X_train_vec.shape[1]}')\n",
    "\n",
    "# Fit Multinomial Naive Bayes\n",
    "model = MultinomialNB()\n",
    "model.fit(X_train_vec, y_train)\n",
    "print('MultinomialNB model training complete!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Activity 7: Model Evaluation\n",
    "- Accuracy, Precision, Recall, and F1-score\n",
    "- Confusion matrix breakdown: True Negatives, False Positives, False Negatives, and True Positives"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = model.predict(X_test_vec)\n",
    "\n",
    "acc = accuracy_score(y_test, y_pred)\n",
    "prec = precision_score(y_test, y_pred)\n",
    "rec = recall_score(y_test, y_pred)\n",
    "f1 = f1_score(y_test, y_pred)\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "tn, fp, fn, tp = cm.ravel()\n",
    "\n",
    "print(f'Accuracy:  {acc*100:.2f}%')\n",
    "print(f'Precision: {prec*100:.2f}%')\n",
    "print(f'Recall:    {rec*100:.2f}%')\n",
    "print(f'F1-Score:  {f1*100:.2f}%')\n",
    "print('\\n' + '='*50)\n",
    "print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))\n",
    "print('='*50)\n",
    "print(f'True Negatives (TN):  {tn} (Ham correctly classified)')\n",
    "print(f'False Positives (FP): {fp} (Ham incorrectly classified as Spam)')\n",
    "print(f'False Negatives (FN): {fn} (Spam incorrectly classified as Genuine)')\n",
    "print(f'True Positives (TP):  {tp} (Spam correctly classified)')\n",
    "\n",
    "# Plot Confusion Matrix\n",
    "plt.figure(figsize=(5.5, 4.5))\n",
    "plt.imshow(cm, cmap=plt.cm.Blues)\n",
    "plt.title('Confusion Matrix - Multinomial Naive Bayes', fontweight='bold', pad=12)\n",
    "plt.colorbar()\n",
    "plt.xticks([0, 1], ['Predicted Ham', 'Predicted Spam'])\n",
    "plt.yticks([0, 1], ['Actual Ham', 'Actual Spam'])\n",
    "for i in range(2):\n",
    "    for j in range(2):\n",
    "        val = cm[i, j]\n",
    "        tag = 'TN' if (i==0 and j==0) else ('FP' if (i==0 and j==1) else ('FN' if (i==1 and j==0) else 'TP'))\n",
    "        plt.text(j, i, f'{val}\\n({tag})', ha='center', va='center', color='white' if val > cm.max()/2 else 'black', fontweight='bold')\n",
    "plt.xlabel('Predicted Label', fontweight='bold')\n",
    "plt.ylabel('Actual Label', fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Activity 8: Real-Time Testing & Inference\n",
    "- Classify incoming unseen SMS messages\n",
    "- Output predicted class along with posterior probabilities: $P(\\text{Spam} \\mid \\text{Message})$ and $P(\\text{Ham} \\mid \\text{Message})$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_messages = [\n",
    "    'Congratulations! You have won a free prize. Claim now.',\n",
    "    'Hey bro, are we meeting at the campus library this evening?',\n",
    "    'URGENT! Your mobile number won 1000 prize. Call now to claim offer.',\n",
    "    'Can you send me the lecture notes for probability and statistics?'\n",
    "]\n",
    "\n",
    "for msg in test_messages:\n",
    "    vec = vectorizer.transform([msg])\n",
    "    pred = model.predict(vec)[0]\n",
    "    probs = model.predict_proba(vec)[0]\n",
    "    \n",
    "    print(f'Incoming SMS: \"{msg}\"')\n",
    "    print(f'Prediction:   {\"Spam\" if pred == 1 else \"Ham (Genuine)\"}')\n",
    "    print(f'P(Spam | Message) = {probs[1]:.4f}')\n",
    "    print(f'P(Ham | Message)  = {probs[0]:.4f}')\n",
    "    print('-' * 60)"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("sms_spam_detection.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1)

print("[+] sms_spam_detection.ipynb successfully generated!")
