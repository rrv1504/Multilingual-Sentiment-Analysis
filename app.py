import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sentiment_utils import clean_text


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer


@st.cache_data
def load_data():
    df = pd.read_csv("amazon_style_reviews.csv")
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    return df


def predict_reviews(reviews):
    cleaned_reviews = [clean_text(review) for review in reviews]
    vectors = vectorizer.transform(cleaned_reviews)
    predictions = model.predict(vectors)
    confidence_scores = model.predict_proba(vectors).max(axis=1) * 100

    return pd.DataFrame({
        "review": reviews,
        "predicted_sentiment": predictions,
        "confidence": confidence_scores.round(1),
    })


def sentiment_chart_data(df, sentiment_col):
    return df[sentiment_col].value_counts().rename_axis("sentiment").reset_index(name="count")


def grouped_sentiment_chart_data(df, group_col, sentiment_col):
    return df.groupby([group_col, sentiment_col]).size().unstack(fill_value=0)


def sentiment_color(sentiment):
    colors = {
        "positive": "#2e7d32",
        "negative": "#c62828",
        "neutral": "#1565c0",
    }
    return colors.get(str(sentiment).lower(), "#546e7a")


def show_sentiment_count_chart(data):
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [sentiment_color(sentiment) for sentiment in data.index]

    ax.bar(data.index.astype(str), data["count"], width=0.35, color=colors)

    ax.set_title("Positive vs Negative Count", fontsize=13, pad=12)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Reviews")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_thin_bar_chart(data, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [sentiment_color(column) for column in data.columns]

    data.plot(kind="bar", ax=ax, width=0.45, color=colors)

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelrotation=25)
    ax.legend(title="Sentiment", frameon=False)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_monthly_trend(data):
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#2e7d32", "#c62828", "#1565c0", "#ef6c00"]

    data.plot(ax=ax, marker="o", linewidth=2, color=colors[: len(data.columns)])

    ax.set_title("Monthly Sentiment Trend", fontsize=13, pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Reviews")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Sentiment", frameon=False)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


model, vectorizer = load_model()

# ---------------- UI ----------------
st.title("Product Review Sentiment Analysis")

st.write("Analyze customer reviews using Machine Learning")

if "review_history" not in st.session_state:
    st.session_state.review_history = []

# ---------------- SINGLE REVIEW ----------------
st.subheader("Single Review Analysis")

user_input = st.text_area("Enter your review:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a review")
    else:
        result = predict_reviews([user_input])
        pred = result.loc[0, "predicted_sentiment"]
        confidence = result.loc[0, "confidence"]

        if pred == "positive":
            st.success(f"Positive Review ({confidence:.1f}% confidence)")
        else:
            st.error(f"Negative Review ({confidence:.1f}% confidence)")

        st.session_state.review_history.insert(0, {
            "review": user_input,
            "predicted_sentiment": pred,
            "confidence": confidence,
        })

# ---------------- REVIEW HISTORY ----------------
st.subheader("Review History")

if st.session_state.review_history:
    history_df = pd.DataFrame(st.session_state.review_history)
    st.dataframe(history_df, width="stretch")

    if st.button("Clear History"):
        st.session_state.review_history = []
        st.rerun()
else:
    st.info("No reviews analyzed yet.")

# ---------------- CSV UPLOAD ----------------
st.subheader("Bulk CSV Analysis")

uploaded_file = st.file_uploader("Upload a CSV file with a review column", type=["csv"])

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview")
    st.dataframe(uploaded_df.head(), width="stretch")

    review_columns = [col for col in uploaded_df.columns if col.lower() in ["review", "reviews", "text", "comment"]]

    if not review_columns:
        st.warning("CSV must contain a review column. Accepted names: review, reviews, text, comment.")
    else:
        selected_review_col = st.selectbox("Select review column", review_columns)

        if st.button("Analyze Uploaded CSV"):
            valid_reviews = uploaded_df[selected_review_col].fillna("").astype(str)
            bulk_results = predict_reviews(valid_reviews)
            result_df = uploaded_df.copy()
            result_df["predicted_sentiment"] = bulk_results["predicted_sentiment"]
            result_df["confidence"] = bulk_results["confidence"]

            st.session_state.bulk_results = result_df

if "bulk_results" in st.session_state:
    st.write("Bulk Analysis Results")

    bulk_counts = st.session_state.bulk_results["predicted_sentiment"].value_counts()
    positive_count = int(bulk_counts.get("positive", 0))
    negative_count = int(bulk_counts.get("negative", 0))

    col1, col2 = st.columns(2)
    col1.metric("Positive Reviews", positive_count)
    col2.metric("Negative Reviews", negative_count)

    st.dataframe(st.session_state.bulk_results, width="stretch")

    csv_data = st.session_state.bulk_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Results CSV",
        data=csv_data,
        file_name="sentiment_analysis_results.csv",
        mime="text/csv",
    )

# # ---------------- DATASET INSIGHTS ----------------
# st.subheader("Dataset Insights")

df = load_data()

# total = len(df)
# counts = df["label"].value_counts()

# positive = counts.get("positive", 0)
# negative = counts.get("negative", 0)

# st.write(f"Total Reviews: {total}")
# if total:
#     st.write(f"Positive: {round(positive / total * 100, 2)}%")
#     st.write(f"Negative: {round(negative / total * 100, 2)}%")

# ---------------- GRAPHS ----------------
st.subheader("Graphs")

st.write("Positive vs Negative Count")
count_data = sentiment_chart_data(df, "label").set_index("sentiment")
show_sentiment_count_chart(count_data)

st.write("Category-wise Sentiment")
if "category" in df.columns:
    category_data = grouped_sentiment_chart_data(df, "category", "label")
    show_thin_bar_chart(category_data, "Category-wise Sentiment", "Category", "Reviews")
    st.dataframe(category_data, width="stretch")
else:
    st.info("No category column found in the dataset.")

st.write("Rating-wise Sentiment")
if "rating" in df.columns:
    rating_data = grouped_sentiment_chart_data(df, "rating", "label")
    show_thin_bar_chart(rating_data, "Rating-wise Sentiment", "Rating", "Reviews")
    st.dataframe(rating_data, width="stretch")
else:
    st.info("No rating column found in the dataset.")

st.write("Monthly Sentiment Trend")
if "date" in df.columns:
    trend_df = df.copy()
    trend_df["date"] = pd.to_datetime(trend_df["date"], dayfirst=True, errors="coerce")
    trend_df = trend_df.dropna(subset=["date"])

    if trend_df.empty:
        st.info("No valid dates found for monthly trend.")
    else:
        trend_df["month"] = trend_df["date"].dt.to_period("M").dt.to_timestamp()
        monthly_data = grouped_sentiment_chart_data(trend_df, "month", "label").sort_index()
        show_monthly_trend(monthly_data)
        st.dataframe(monthly_data, width="stretch")
else:
    st.info("No date column found in the dataset.")
