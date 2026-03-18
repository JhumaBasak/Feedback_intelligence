import streamlit as st
import pandas as pd
from model import train_model

# Load model
model, vectorizer, accuracy = train_model()

# UI
st.set_page_config(page_title="Feedback Intelligence Platform", layout="wide")

st.title("🚀 Customer Feedback Intelligence Platform")
st.markdown("### ML-powered analytics for business insights")

# Sidebar
st.sidebar.header("Model Info")
st.sidebar.write(f"Model Accuracy: {accuracy:.2f}")

# Upload Section
st.header("📂 Upload Dataset")
file = st.file_uploader("Upload CSV with 'review' column", type=["csv"])

if file:
    df = pd.read_csv(file)

    if "review" not in df.columns:
        st.error("CSV must contain 'review' column")
    else:
        vectors = vectorizer.transform(df["review"])
        preds = model.predict(vectors)
        probs = model.predict_proba(vectors).max(axis=1)

        df["Category"] = preds
        df["Confidence"] = probs

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Category Distribution")
            st.bar_chart(df["Category"].value_counts())

        with col2:
            st.subheader("📈 Confidence Distribution")
            st.line_chart(df["Confidence"])

        st.subheader("📋 Data Preview")
        st.dataframe(df)

        # Download button
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results", csv, "results.csv")

# Single Input
st.header("🔍 Real-Time Analysis")

review = st.text_area("Enter customer review")

if st.button("Analyze"):
    vec = vectorizer.transform([review])
    pred = model.predict(vec)[0]
    conf = model.predict_proba(vec).max()

    st.success(f"Category: {pred}")
    st.info(f"Confidence: {conf:.2f}")

    insights = {
        "Delivery": "Improve logistics efficiency",
        "Product": "Focus on product quality",
        "Service": "Enhance support experience",
        "Pricing": "Re-evaluate pricing strategy"
    }

    st.write(f"💡 Insight: {insights.get(pred)}")