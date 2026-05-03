# 📊 Multilingual Sentiment Analysis System

A **Machine Learning-based web application** that analyzes customer reviews and predicts whether they are **Positive or Negative**.
This project supports **multilingual input** including **English, Hindi, and Gujarati**.

---

## 🚀 Features

* 🔍 **Single Review Analysis**

  * Enter a review and instantly get sentiment prediction with confidence score

* 📂 **Bulk CSV Analysis**

  * Upload a CSV file and analyze multiple reviews at once
  * Download results with predictions

* 🌐 **Multilingual Support**

  * Works with English, Hindi, Gujarati, and mixed (Hinglish/Gujlish) text

* 📊 **Interactive Visualizations**

  * Positive vs Negative count
  * Category-wise sentiment
  * Rating-wise sentiment
  * Monthly trend analysis

* 🧠 **Machine Learning Model**

  * Uses TF-IDF + Logistic Regression for prediction

---

## 🏗️ Project Structure

```
.
├── app.py                 # Streamlit web app :contentReference[oaicite:0]{index=0}
├── model.py               # Model training script :contentReference[oaicite:1]{index=1}
├── model.pkl              # Trained ML model
├── vectorizer.pkl         # TF-IDF vectorizer
├── sentiment_utils.py     # Text preprocessing + multilingual logic :contentReference[oaicite:2]{index=2}
├── amazon_style_reviews.csv # Dataset
├── reviews.csv            # Custom dataset
├── requirements.txt       # Dependencies :contentReference[oaicite:3]{index=3}
└── README.md              # Project documentation
```

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone <your-repo-link>
cd sentiment-analysis
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The app will open in your browser.

---

## 🧠 Model Details

* **Algorithm:** Logistic Regression
* **Vectorization:** TF-IDF (Unigrams + Bigrams)
* **Features:** Max 1000 features
* **Handling Imbalance:** `class_weight="balanced"`
* **Train-Test Split:** 80-20

Model training and saving is handled in `model.py`.

---

## 🌍 Multilingual Processing

The system uses custom logic from `sentiment_utils.py`:

* Detects **Hindi & Gujarati sentiment keywords**
* Converts them into **English sentiment tokens**
* Improves prediction accuracy for mixed-language input

Example:

```
"આ પ્રોડક્ટ સરસ છે" → Positive
"यह खराब है" → Negative
```

---

## 📂 Dataset Format

CSV file must contain a review column:

```
review,label
"Great product",positive
"Very खराब experience",negative
```

Accepted column names:

* review
* reviews
* text
* comment

---

## 📊 Output Example

| Review        | Predicted Sentiment | Confidence |
| ------------- | ------------------- | ---------- |
| Great product | Positive            | 92.3%      |
| Very खराब     | Negative            | 88.7%      |

---

## 📌 Dependencies

* streamlit
* pandas
* scikit-learn
* matplotlib

(Defined in `requirements.txt`)

---

## ⚠️ Limitations

* Only supports **binary classification (Positive / Negative)**
* Accuracy depends on dataset quality
* Rule-based multilingual support (not deep NLP)

---

## 🔮 Future Improvements

* Add **Neutral sentiment**
* Use **Deep Learning (LSTM / BERT)**
* Improve multilingual handling with **transformers**
* Deploy on cloud (Streamlit Cloud / AWS)

---

## 👩‍💻 Author

**Sakshi S. Doshi**
**Mahi Shah**
**Kamya Shah**
**Roshni Raichandani**


---

## ⭐ Conclusion

This project demonstrates how **Machine Learning + NLP** can be used to build a **real-world sentiment analysis system** with multilingual capabilities.
