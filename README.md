# Toxic Comment Detection App

This repository contains two Streamlit apps:

## 🚀 Apps

- `user.py`: A user-facing app for submitting comments and checking toxicity.
- `admin.py`: An admin dashboard for analyzing, moderating, and exporting comments.

## 🧠 ML Model

- Model: Logistic Regression with TF-IDF
- Files:
  - `toxic_comment_model.pkl`: Trained model
  - `tfidf_vectorizer.pkl`: Vectorizer

## 📝 Features

### User App:
- Comment submission.
- Detect if comment is toxic.
- Edit toxic comments or submit anyway.
- View recent comments.

### Admin App:
- View total, toxic, and non-toxic comment stats.
- Pie chart of toxicity distribution.
- Word cloud of toxic comments.
- User activity report.
- Download comments as CSV.

## 🔧 Installation

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt

Run User App:
streamlit run user.py
Run Admin App:
streamlit run admin.py
