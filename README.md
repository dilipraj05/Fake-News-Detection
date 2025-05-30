## 📰 Fake News Detection Using Machine Learning

### 📌 Project Summary

This project aims to detect whether a news article is real or fake using Natural Language Processing (NLP) and machine learning. We use TF-IDF vectorization and a Logistic Regression model for classification.

---

### 📂 Folder Structure

```
Fake-News-Detection/
├── data/
│   ├── fake.csv
│   └── real.csv
├── notebooks/
│   └── fake_news_analysis.ipynb
├── models/
│   └── logistic_model.pkl
├── outputs/
│   ├── wordclouds/
│   └── confusion_matrix.png
├── app/
│   └── streamlit_app.py  # Optional
├── README.md
└── requirements.txt
```

---

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

### 🔧 Steps Performed

1. **Data Loading**: Combined real and fake news with appropriate labels.
2. **EDA**: Analyzed article length, text distribution, and generated word clouds.
3. **Text Cleaning**: Lowercasing, punctuation removal, stopword removal.
4. **Feature Extraction**: TF-IDF vectorization.
5. **Model Building**: Logistic Regression.
6. **Evaluation**: Confusion matrix, accuracy, precision, recall, F1-score.
7. **(Optional)**: UI for live news prediction with Streamlit.

---

### 📈 Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.95+ |
| Precision | 0.95+ |
| Recall    | 0.95+ |
| F1-Score  | 0.95+ |

---

### ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/dilipraj05/fake-news-detection.git
cd fake-news-detection

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/fake_news_analysis.ipynb
```

To run the optional Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

---

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
