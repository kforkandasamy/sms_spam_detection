import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "message"], header=None)
df = df.drop_duplicates().reset_index(drop=True)
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

X = df["message"]
y = df["label_num"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
vectorizer = CountVectorizer(stop_words="english", lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Export vocabulary and feature log probs for client-side evaluation
vocab = vectorizer.vocabulary_
# model.feature_log_prob_[0] is Ham, model.feature_log_prob_[1] is Spam
feature_log_prob_ham = model.feature_log_prob_[0].tolist()
feature_log_prob_spam = model.feature_log_prob_[1].tolist()
class_log_prior = model.class_log_prior_.tolist()

export_data = {
    "vocabulary": vocab,
    "log_prior_ham": class_log_prior[0],
    "log_prior_spam": class_log_prior[1],
    "log_prob_ham": feature_log_prob_ham,
    "log_prob_spam": feature_log_prob_spam
}

with open("model_export.json", "w") as f:
    json.dump(export_data, f)

print("[+] model_export.json created successfully!")
