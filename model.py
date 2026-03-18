import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model():
    data = {
        "review": [
            "The product quality is excellent",
            "Delivery was very late",
            "Customer service was very helpful",
            "Price is too high",
            "Great product and quality",
            "Late delivery and poor service",
            "Affordable pricing and good value",
            "Support team resolved my issue quickly"
        ],
        "category": [
            "Product",
            "Delivery",
            "Service",
            "Pricing",
            "Product",
            "Delivery",
            "Pricing",
            "Service"
        ]
    }

    df = pd.DataFrame(data)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["review"])
    y = df["category"]

    model = LogisticRegression()
    model.fit(X, y)

    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    return model, vectorizer, acc