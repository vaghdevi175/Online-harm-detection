import streamlit as st
import pandas as pd
import os
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ---------------- Load Comments ----------------
def load_comments_data(filename="submitted_comments.csv"):
    try:
        if not os.path.exists(filename):
            st.warning("No comments data found.")
            return pd.DataFrame(columns=["comment", "username", "timestamp", "profile_color", "avatar", "is_toxic"])

        df = pd.read_csv(filename)

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        else:
            df['timestamp'] = pd.NaT

        if 'is_toxic' not in df.columns or df['is_toxic'].isna().any():
            df['is_toxic'] = df['comment'].apply(detect_toxic_comments)

        df['is_toxic'] = df['is_toxic'].fillna(False).astype(bool)
        return df

    except Exception as e:
        st.error(f"Error loading comments: {e}")
        return pd.DataFrame(columns=["comment", "username", "timestamp", "profile_color", "avatar", "is_toxic"])


# ---------------- Detect Toxic Comments ----------------
def detect_toxic_comments(comment):
    try:
        if pd.isna(comment) or comment == "":
            return False

        tfidf = joblib.load('tfidf_vectorizer.pkl')
        model = joblib.load('toxic_comment_model.pkl')
        features = tfidf.transform([comment])
        prediction = model.predict(features)
        return bool(prediction[0])
    except Exception as e:
        st.error(f"Error in detecting toxicity: {e}")
        return False


# ---------------- Save Comments ----------------
def save_comment_to_csv(comment, username, profile_color, avatar, is_toxic, filename="submitted_comments.csv"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_comment = pd.DataFrame([{
        "comment": comment,
        "username": username,
        "timestamp": timestamp,
        "profile_color": profile_color,
        "avatar": avatar,
        "is_toxic": is_toxic
    }])

    if os.path.exists(filename):
        comments_df = pd.read_csv(filename)
        comments_df = pd.concat([comments_df, new_comment], ignore_index=True)
    else:
        comments_df = new_comment

    comments_df.to_csv(filename, index=False)


# ---------------- Train Model ----------------
def train_model(dataset_path):
    data = pd.read_csv(dataset_path)

    if 'text' not in data.columns or 'label' not in data.columns:
        st.error("Dataset must contain 'text' and 'label' columns.")
        return

    data.dropna(subset=['text', 'label'], inplace=True)

    X = data['text']
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    tfidf = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.text("Classification Report:\n")
    st.text(classification_report(y_test, y_pred))

    joblib.dump(model, 'toxic_comment_model.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    st.success("Model and vectorizer saved!")


# ---------------- Admin Dashboard ----------------
def admin_dashboard():
    st.title('🛠️ Admin Dashboard')
    comments_df = load_comments_data()

    page = st.sidebar.selectbox(
        'Navigate',
        ['Overview', 'Toxic Comment Analysis', 'User Management']
    )

    if page == 'Overview':
        st.header('Comments Overview')
        total_comments = len(comments_df)
        toxic_comments = comments_df['is_toxic'].sum()
        non_toxic_comments = total_comments - toxic_comments

        col1, col2, col3 = st.columns(3)
        col1.metric('Total Comments', total_comments)
        col2.metric('Toxic Comments', toxic_comments)
        col3.metric('Non-Toxic Comments', non_toxic_comments)

        st.subheader('Toxicity Distribution')
        if total_comments > 0:
            labels = ['Toxic Comments', 'Non-Toxic Comments']
            sizes = [toxic_comments, non_toxic_comments]
            colors = ['#ff6347', '#90ee90']
            explode = (0.1, 0)

            fig, ax = plt.subplots()
            ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=140)
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.info("No comments available to display pie chart.")

        st.subheader('Recent Comments')
        if not comments_df.empty:
            recent_comments = comments_df.sort_values(
                'timestamp', ascending=False).head(100)
            st.dataframe(recent_comments)
        else:
            st.info('No comments to display.')

    elif page == 'Toxic Comment Analysis':
        st.header('Toxic Comment Analysis')
        toxic_comments = comments_df[comments_df['is_toxic']]

        if not toxic_comments.empty:
            st.dataframe(toxic_comments)
        else:
            st.info('No toxic comments detected.')

    elif page == 'User Management':
        if not comments_df.empty:
            user_activity = comments_df.groupby('username').agg(
                Total_Comments=('comment', 'count'),
                Toxic_Comments=('is_toxic', 'sum')
            ).reset_index()
            st.subheader('User Activity Report')
            st.dataframe(user_activity)
        else:
            st.info('No user data available.')

    st.sidebar.markdown("---")


# ---------------- Main Function ----------------
def main():
    st.set_page_config(page_title='Admin Dashboard', layout='wide')

    if 'admin_logged_in' not in st.session_state:
        st.session_state.admin_logged_in = False

    if not st.session_state.admin_logged_in:
        st.title('🔐 Admin Login')
        username = st.text_input('Username')
        password = st.text_input('Password', type='password')

        if st.button('Login'):
            if username == 'admin' and password == '123':
                st.session_state.admin_logged_in = True
                st.success('Login Successful!')
                st.experimental_rerun()

            else:
                st.error('Invalid Credentials')
    else:
        admin_dashboard()


if __name__ == '__main__':
    main()
