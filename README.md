## 📰 Fake News Detection Using Machine Learning

### 📌 Project Summary

This project aims to detect whether a news article is real or fake using Natural Language Processing (NLP) and machine learning. We use TF-IDF vectorization and a Logistic Regression model for classification.

### 📊 Technologies Used

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* WordCloud
* Scikit-learn
* NLTK
* Streamlit (optional UI)

---

### 🔍 Dataset

* Source: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
* `fake.csv`: 23,481 fake news articles
* `real.csv`: 21,417 real news articles

---

## 🎯 Objective

To analyze news data and build a model that can classify whether a news article is **real** or **fake**, showcasing data analysis, visualization, and basic NLP skills.

---

## 📊 Exploratory Data Analysis (EDA)

- News class distribution (Fake vs Real)
- WordClouds for fake and real news
- Top keywords for each class
- Sentiment comparison
- Misclassification analysis

![EDA Sample](https://via.placeholder.com/400x200.png?text=Add+Your+Own+EDA+Graph+Here)

---

## 🧹 Text Preprocessing

- Lowercasing, punctuation & stopword removal
- Combined title + text
- Vectorized using **TF-IDF**

---

## 🤖 Model Used

- Logistic Regression (Lightweight and accurate)
- Accuracy: **~94%**
- Confusion Matrix, Precision, Recall & F1 Score

---

## 🧠 Tools & Libraries

- Python
- Pandas, Numpy
- Matplotlib, Seaborn
- Scikit-learn
- WordCloud
- Streamlit (optional UI)

---

## 📈 Results

| Metric     | Value     |
|------------|-----------|
| Accuracy   | 94%       |
| Precision  | 93%       |
| Recall     | 95%       |
| F1 Score   | 94%       |

---

## 🖥️ Optional: Streamlit App

Run this command:
```bash
streamlit run app.py

### 📌 Future Improvements

* Use deep learning models (LSTM, BERT)
* Add more features like title/source
* Expand language support
* Deploy using HuggingFace or Render

---

### 🙋‍♂️ Author

**Dilip Raj**
[GitHub: dilipraj05](https://github.com/dilipraj05)

---
