import streamlit as st
import pandas as pd
import random
import datetime
import joblib
import os


# ================= Generate Random User ================= #
def generate_user_profile():
    usernames = [
        "Tech Enthusiast", "Code Ninja", "Digital Explorer",
        "Cyber Wizard", "Data Detective", "Innovation Guru",
        "Tech Maverick", "Pixel Pioneer", "Coding Champion"
    ]
    r, g, b = random.randint(100, 200), random.randint(100, 200), random.randint(100, 200)
    return {
        'username': random.choice(usernames),
        'profile_color': f'rgb({r},{g},{b})',
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


# ================= Load Model ================= #
def load_model():
    try:
        tfidf = joblib.load('tfidf_vectorizer.pkl')
        model = joblib.load('toxic_comment_model.pkl')
        return tfidf, model
    except Exception as e:
        st.error(f"Model not found. Error: {e}")
        st.stop()


# ================= Predict Toxicity ================= #
def predict_toxicity(comment, tfidf, model):
    comment_tfidf = tfidf.transform([comment])
    prediction = model.predict(comment_tfidf)[0]
    return prediction == 1


# ================= Save to CSV ================= #
def save_comment_to_csv(comment, username, profile_color, timestamp, is_toxic, filename="submitted_comments.csv"):
    new_comment = pd.DataFrame([{
        "comment": comment,
        "username": username,
        "timestamp": timestamp,
        "profile_color": profile_color,
        "avatar": "",  # Optional, can be left blank
        "is_toxic": is_toxic
    }])

    if os.path.exists(filename):
        existing_comments = pd.read_csv(filename)
        updated_comments = pd.concat([existing_comments, new_comment], ignore_index=True)
    else:
        updated_comments = new_comment

    updated_comments.to_csv(filename, index=False, encoding='utf-8')


# ================= Format Timestamp ================= #
def format_timestamp(timestamp):
    try:
        timestamp = pd.to_datetime(timestamp)
    except:
        return "Unknown time"

    now = datetime.datetime.now()
    diff = now - timestamp

    if diff.days > 365:
        return f"{diff.days // 365} years ago"
    elif diff.days > 30:
        return f"{diff.days // 30} months ago"
    elif diff.days > 0:
        return f"{diff.days} days ago"
    elif diff.seconds > 3600:
        return f"{diff.seconds // 3600} hours ago"
    elif diff.seconds > 60:
        return f"{diff.seconds // 60} minutes ago"
    else:
        return "just now"


# ================= Main App ================= #
def main():
    st.set_page_config(page_title="Toxic Comment Detector", page_icon="💬")
    st.title("💬 Toxic Comment Detector")

    # CSS for styling
    st.markdown("""
    <style>
    .comment-container {
        display: flex;
        align-items: flex-start;
        margin-bottom: 15px;
        padding: 10px;
        background-color: #f8f8f8;
        border-radius: 10px;
    }
    .profile-pic {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    }
    .comment-content {
        flex-grow: 1;
    }
    .comment-header {
        display: flex;
        align-items: center;
        margin-bottom: 5px;
    }
    .username {
        font-weight: bold;
        margin-right: 10px;
    }
    .timestamp {
        color: #606060;
        font-size: 0.8em;
    }
    </style>
    """, unsafe_allow_html=True)

    # Load ML model
    tfidf, model = load_model()

    # Session state initialization
    if 'submitted_comments' not in st.session_state:
        st.session_state.submitted_comments = []

    if 'toxic_comment' not in st.session_state:
        st.session_state.toxic_comment = None

    st.header("✍️ Enter Your Comment")

    # If previous toxic comment detected
    if st.session_state.toxic_comment:
        st.warning("⚠️ Your previous comment may contain offensive content.")
        edited_comment = st.text_area("Edit your comment:", value=st.session_state.toxic_comment, height=150)

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Re-check Comment"):
                is_toxic = predict_toxicity(edited_comment, tfidf, model)
                if is_toxic:
                    st.error("The edited comment is still flagged as toxic.")
                else:
                    user_profile = generate_user_profile()
                    comment_data = {
                        'comment': edited_comment,
                        'username': user_profile['username'],
                        'timestamp': user_profile['timestamp'],
                        'profile_color': user_profile['profile_color']
                    }
                    st.session_state.submitted_comments.append(comment_data)

                    save_comment_to_csv(
                        edited_comment,
                        user_profile['username'],
                        user_profile['profile_color'],
                        user_profile['timestamp'],
                        is_toxic=False
                    )

                    st.session_state.toxic_comment = None
                    st.rerun()

        with col2:
            if st.button("Submit Anyway"):
                user_profile = generate_user_profile()
                comment_data = {
                    'comment': st.session_state.toxic_comment,
                    'username': user_profile['username'],
                    'timestamp': user_profile['timestamp'],
                    'profile_color': user_profile['profile_color']
                }
                st.session_state.submitted_comments.append(comment_data)

                save_comment_to_csv(
                    st.session_state.toxic_comment,
                    user_profile['username'],
                    user_profile['profile_color'],
                    user_profile['timestamp'],
                    is_toxic=True
                )

                st.session_state.toxic_comment = None
                st.rerun()

        with col3:
            if st.button("Cancel"):
                st.session_state.toxic_comment = None
                st.rerun()

    else:
        comment = st.text_area("", height=150)

        if st.button("Submit"):
            if not comment.strip():
                st.warning("Please enter a comment.")
            else:
                is_toxic = predict_toxicity(comment, tfidf, model)
                if is_toxic:
                    st.session_state.toxic_comment = comment
                    st.rerun()
                else:
                    user_profile = generate_user_profile()
                    comment_data = {
                        'comment': comment,
                        'username': user_profile['username'],
                        'timestamp': user_profile['timestamp'],
                        'profile_color': user_profile['profile_color']
                    }
                    st.session_state.submitted_comments.append(comment_data)

                    save_comment_to_csv(
                        comment,
                        user_profile['username'],
                        user_profile['profile_color'],
                        user_profile['timestamp'],
                        is_toxic=False
                    )

                    st.success("✅ Comment submitted successfully!")

    # Display Comments Feed
    if st.session_state.submitted_comments:
        st.header("🗨️ Comment Feed")
        for comment_data in reversed(st.session_state.submitted_comments):
            comment = comment_data['comment']
            profile = comment_data['username']
            profile_color = comment_data['profile_color']
            timestamp = format_timestamp(comment_data['timestamp'])
            comment_html = f"""
                <div class="comment-container">
                    <div class="profile-pic" style="background-color: {profile_color};">
                        {profile[0]}
                    </div>
                    <div class="comment-content">
                        <div class="comment-header">
                            <span class="username">{profile}</span>
                            <span class="timestamp">{timestamp}</span>
                        </div>
                        <div class="comment-text">{comment}</div>
                    </div>
                </div>
            """
            st.markdown(comment_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
