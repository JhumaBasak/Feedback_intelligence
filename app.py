import streamlit as st
import pandas as pd
import joblib
import hashlib
import datetime
import altair as alt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =====================================================
# Utility Functions
# =====================================================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# =====================================================
# Session State Initialization
# =====================================================
if "users" not in st.session_state:
    st.session_state["users"] = {}
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
for key in ["model", "vectorizer", "accuracy", "last_trained", "uploaded_file"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "retrain_history" not in st.session_state:
    st.session_state["retrain_history"] = []

# =====================================================
# Authentication Functions
# =====================================================
def signup_user(username, password):
    if username in st.session_state["users"]:
        return False, "❌ Username already exists"
    if not username.strip() or not password.strip():
        return False, "⚠️ Username and password cannot be empty"
    st.session_state["users"][username] = hash_password(password)
    return True, "✅ Signup successful! Please log in."

def login_user(username, password):
    if username in st.session_state["users"] and st.session_state["users"][username] == hash_password(password):
        st.session_state["authenticated"] = True
        return True, f"🎉 Welcome {username}!"
    return False, "❌ Invalid username or password"

def logout():
    st.session_state["authenticated"] = False
    st.rerun()

# =====================================================
# Model Training
# =====================================================
def train_and_save_model():
    df = pd.read_csv("training_data.csv")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X = df["review"]
    y = df["category"]

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    joblib.dump(acc, "accuracy.pkl")
    joblib.dump(timestamp, "last_trained.pkl")

    return model, vectorizer, acc, prec, rec, f1, timestamp

# Load or retrain
try:
    st.session_state["model"] = joblib.load("model.pkl")
    st.session_state["vectorizer"] = joblib.load("vectorizer.pkl")
    st.session_state["accuracy"] = joblib.load("accuracy.pkl")
    st.session_state["last_trained"] = joblib.load("last_trained.pkl")
except Exception:
    m, v, a, p, r, f, t = train_and_save_model()
    st.session_state["model"], st.session_state["vectorizer"], st.session_state["accuracy"], st.session_state["last_trained"] = m, v, a, t

# =====================================================
# Pages
# =====================================================
def signup_page():
    st.markdown("<h1 style='color: teal;'>📝 Signup</h1>", unsafe_allow_html=True)
    username = st.text_input("Create Username", key="signup_username")
    password = st.text_input("Create Password", type="password", key="signup_password")
    if st.button("Signup"):
        success, message = signup_user(username, password)
        if success:
            st.success(message)
        else:
            st.error(message)

def login_page():
    st.markdown("<h1 style='color: teal;'>🔑 Login</h1>", unsafe_allow_html=True)
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        success, message = login_user(username, password)
        if success:
            st.success(message)
            st.rerun()
        else:
            st.error(message)

def app_page():
    st.markdown("<h1 style='color: darkblue;'>📊 Customer Feedback Analysis Dashboard</h1>", unsafe_allow_html=True)

    # 🎨 Accuracy Card
    st.markdown(
        f"""
        <div style="background-color:#4CAF50;padding:15px;border-radius:10px;margin-bottom:15px">
            <h2 style="color:white;margin:0;">✨ Accuracy: {st.session_state['accuracy']:.2f}</h2>
            <p style="color:white;margin:0;">🕒 Last trained: {st.session_state['last_trained']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 🔄 Retrain Section
    if st.button("⚙️ Retrain Model Now"):
        st.cache_resource.clear()
        with st.spinner("🔄 Training in progress..."):
            m, v, a, p, r, f, t = train_and_save_model()
        st.session_state["model"] = m
        st.session_state["vectorizer"] = v
        st.session_state["accuracy"] = a
        st.session_state["last_trained"] = t

        # Append to retrain history
        st.session_state["retrain_history"].append({
            "accuracy": a,
            "precision": p,
            "recall": r,
            "f1_score": f,
            "timestamp": datetime.datetime.now()
        })

        st.success("✅ Model retrained successfully!")

    # 📂 File upload + results
    uploaded_file = st.file_uploader("Upload a CSV/Excel file with a 'review' column", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            if "review" not in df.columns:
                st.error("❌ File must contain a 'review' column")
            else:
                vecs = st.session_state["vectorizer"].transform(df["review"].astype(str))
                preds = st.session_state["model"].predict(vecs)
                confs = st.session_state["model"].predict_proba(vecs).max(axis=1)

                results = pd.DataFrame({
                    "Review": df["review"],
                    "Predicted Category": preds,
                    "Confidence": confs
                })

                st.success("✅ File processed successfully!")
                st.dataframe(results, use_container_width=True)

                # 📊 Visualizations
                st.subheader("📈 Category Distribution")
                st.bar_chart(results["Predicted Category"].value_counts())

                st.subheader("📉 Confidence Score Trend")
                st.area_chart(results["Confidence"])

                st.subheader("🧭 Business Insights Summary")
                insights_map = {
                    "Delivery": "🚚 Improve logistics efficiency",
                    "Product": "📦 Focus on product quality",
                    "Service": "🤝 Enhance customer support experience",
                    "Pricing": "💰 Re-evaluate pricing strategy"
                }
                for cat, count in results["Predicted Category"].value_counts().items():
                    st.write(f"**{cat}** ({count} reviews): {insights_map.get(cat, 'General improvement recommended')}")

                # 📥 Download analyzed results
                st.subheader("📥 Export Results")
                csv = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv,
                    file_name="feedback_analysis_results.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")

    # 📜 Retrain History Log
    if st.session_state["retrain_history"]:
        with st.expander("📜 Retrain History"):
            history_df = pd.DataFrame(st.session_state["retrain_history"])
            history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])

            st.dataframe(history_df, use_container_width=True)

            # Melt into long format for Altair
            long_df = history_df.melt(
                id_vars=["timestamp"],
                value_vars=["accuracy", "precision", "recall", "f1_score"],
                var_name="metric",
                value_name="value"
            )

                        # Build Altair chart with custom colors
            chart = alt.Chart(long_df).mark_line(point=True).encode(
                x="timestamp:T",
                y="value:Q",
                color=alt.Color("metric:N",
                                scale=alt.Scale(
                                    domain=["accuracy", "precision", "recall", "f1_score"],
                                    range=["green", "blue", "orange", "purple"]
                                )),
                tooltip=["timestamp:T", "metric:N", "value:Q"]
            ).properties(
                width=700,
                height=400,
                title="Model Performance Metrics Over Time"
            )

            st.altair_chart(chart, use_container_width=True)

# =====================================================
# Sidebar Behavior + Routing
# =====================================================
st.sidebar.title("Menu")

if st.session_state["authenticated"]:
    menu = st.sidebar.radio("Navigation", ["App", "Logout"], index=0)
    if menu == "App":
        app_page()
    elif menu == "Logout":
        logout()
else:
    auth_choice = st.sidebar.radio("Choose an option", ["Signup", "Login"], index=1)
    if auth_choice == "Signup":
        signup_page()
    elif auth_choice == "Login":
        login_page()
