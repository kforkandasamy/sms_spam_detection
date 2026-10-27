import matplotlib.pyplot as plt
import numpy as np
import json
import os

with open("results.json", "r") as f:
    res = json.load(f)

ds = res["dataset"]
words = res["words"]
me = res["model_evaluation"]

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 0.8

# 1. Class Distribution
plt.figure(figsize=(6.5, 4.2), dpi=300)
bars = plt.bar(["Ham (Genuine)", "Spam"], [ds["n_ham"], ds["n_spam"]], 
               color=["#334155", "#64748b"], width=0.42, edgecolor="#0f172a", linewidth=1.2)
plt.title("Class Distribution of Cleaned SMS Messages", fontsize=11, fontweight="bold", pad=12, color="#0f172a")
plt.ylabel("Number of Messages", fontsize=10, color="#334155")
plt.grid(axis="y", linestyle="--", alpha=0.35, color="#94a3b8")
total = ds["total_cleaned"]
for bar in bars:
    yval = bar.get_height()
    pct = (yval / total) * 100
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f"{yval}\n({pct:.1f}%)", 
             ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#0f172a")
plt.ylim(0, max(ds["n_ham"], ds["n_spam"]) * 1.15)
plt.tight_layout()
plt.savefig("plots/class_distribution.png", dpi=300)
plt.close()

# 2. Word Posterior Probabilities
words_list = [w["word"].upper() for w in words]
p_spam = [w["p_spam_given_word_empirical"] for w in words]
p_ham = [1.0 - v for v in p_spam]
x = np.arange(len(words_list))
w_width = 0.35

plt.figure(figsize=(7.5, 4.2), dpi=300)
plt.bar(x - w_width/2, p_spam, w_width, label="P(Spam | Word)", color="#1e293b", edgecolor="#0f172a", linewidth=1)
plt.bar(x + w_width/2, p_ham, w_width, label="P(Ham | Word)", color="#94a3b8", edgecolor="#0f172a", linewidth=1)
plt.xticks(x, words_list, fontweight="bold", fontsize=9.5, color="#0f172a")
plt.ylabel("Posterior Probability", fontsize=10, color="#334155")
plt.title("Posterior Probability of Spam vs Ham Given Suspicious Keywords", fontsize=11, fontweight="bold", pad=12, color="#0f172a")
plt.ylim(0, 1.15)
plt.legend(frameon=True, edgecolor="#cbd5e1", fontsize=9)
plt.grid(axis="y", linestyle="--", alpha=0.35, color="#94a3b8")
for i in range(len(words_list)):
    plt.text(x[i] - w_width/2, p_spam[i] + 0.02, f"{p_spam[i]:.2f}", ha="center", fontsize=8.5, fontweight="bold", color="#0f172a")
    plt.text(x[i] + w_width/2, p_ham[i] + 0.02, f"{p_ham[i]:.2f}", ha="center", fontsize=8.5, fontweight="bold", color="#0f172a")
plt.tight_layout()
plt.savefig("plots/word_probabilities.png", dpi=300)
plt.close()

# 3. Confusion Matrix
cm = np.array([[me["tn"], me["fp"]], [me["fn"], me["tp"]]])
plt.figure(figsize=(5.2, 4.2), dpi=300)
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Greys)
plt.title("Confusion Matrix (Multinomial Naive Bayes)", fontsize=11, fontweight="bold", pad=12, color="#0f172a")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Predicted Ham", "Predicted Spam"], fontsize=9.5, color="#0f172a")
plt.yticks(tick_marks, ["Actual Ham", "Actual Spam"], fontsize=9.5, color="#0f172a")
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        val = cm[i, j]
        label_text = f"{val}\n"
        if i == 0 and j == 0: label_text += "(TN)"
        elif i == 0 and j == 1: label_text += "(FP)"
        elif i == 1 and j == 0: label_text += "(FN)"
        else: label_text += "(TP)"
        plt.text(j, i, label_text, horizontalalignment="center", verticalalignment="center",
                 color="white" if val > thresh else "#0f172a", fontsize=10, fontweight="bold")
plt.ylabel("Actual Label", fontsize=10, color="#334155")
plt.xlabel("Predicted Label", fontsize=10, color="#334155")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png", dpi=300)
plt.close()

print("[+] Clean academic plots regenerated successfully!")
