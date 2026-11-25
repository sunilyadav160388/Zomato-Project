# Zomato Restaurant Clustering & Review Sentiment Analysis

**Capstone Machine Learning Project • Submission Package**

**Candidate Name:** Sunil Yadav
**Organization:** Labmentix
**Project Type:** EDA + Classification (NLP Sentiment Analysis) + Unsupervised Learning (Clustering)
**Contribution:** Individual
**GitHub Link:** https://github.com/sunilyadav160388/Zomato-Project
**Date of Submission:** 25 November 2026

---

## 📌 Executive Summary

This project presents an end-to-end analysis of Zomato restaurants in the Gachibowli area of Hyderabad, using **105 restaurants** (cost for two, cuisines, collections, timings) and **10,000 customer reviews** (100 reviews for each of 100 restaurants). The objectives are to understand what drives customer satisfaction, to automatically classify the sentiment of free-text reviews, and to group restaurants into meaningful segments, so that Zomato can recommend better and restaurant owners can benchmark themselves.

The final model is a **TF-IDF + Logistic Regression** pipeline that classifies a review as Positive (4-5 stars) or Not positive (1-3 stars) with **86.7% accuracy, F1 = 0.894 and ROC-AUC = 0.942**.

---

## 📁 Submission Package Directory Structure

Zomato_Project_Package/
│
├── README.md                                        # Project overview and documentation (this file)
├── Zomato_Restaurant_ML_Project_Report.pdf          # Formal 14-page project report
├── Zomato_Restaurant_ML_Capstone.ipynb              # Fully executed Jupyter Notebook (template format, 15+ charts, 3 models)
├── Zomato_Restaurant_ML_Presentation.pptx           # 14-slide 16:9 presentation with speaker notes
│
├── data/                                            # Input datasets
│   ├── Zomato_Restaurant_names_and_Metadata.csv     # 105 restaurants: cost, cuisines, collections, timings
│   └── Zomato_Restaurant_reviews.csv                # 10,000 customer reviews
│
├── model/
│   └── zomato_sentiment_model.joblib                # Saved final pipeline (TF-IDF + chi-square + Logistic Regression)
│
└── charts/                                          # 29 high-resolution chart exports (PNG)
    ├── 00_missing_values_heatmap.png
    ├── 01_rating_distribution.png
    ├── 02_cost_for_two_distribution.png
    ├── 03_top_10_cuisines.png
    ├── 04_reviews_over_time.png
    ├── 05_review_length_distribution.png
    ├── 06_top_bottom_restaurants.png
    ├── 07_cost_vs_average_rating.png
    ├── 08_review_length_by_rating.png
    ├── 09_pictures_vs_rating.png
    ├── 10_reviews_by_hour.png
    ├── 11_rating_by_weekday.png
    ├── 12_rating_by_cuisine.png
    ├── 13_reviewer_followers_vs_rating.png
    ├── 14_correlation_heatmap.png
    ├── 15_pair_plot.png
    ├── 16_outlier_capping.png
    ├── 17_word_clouds.png
    ├── 18_svd_2d_reviews.png
    ├── 19_lr_base_metrics.png
    ├── 20_lr_tuned_metrics.png
    ├── 21_nb_base_metrics.png
    ├── 22_nb_tuned_metrics.png
    ├── 23_rf_base_metrics.png
    ├── 24_rf_tuned_metrics.png
    ├── 25_model_comparison.png
    ├── 26_kmeans_elbow_silhouette.png
    ├── 27_restaurant_clusters_pca.png
    └── 28_top_words_coefficients.png
```

---

## 🎯 Key Project Highlights & Findings

1. **Polarised Satisfaction:** About 63% of reviews are 4-5 stars (38% are 5 stars), yet 17% are 1 star and only 12% are 3 stars. Customers either love or dislike the experience, so the 1-star group should be investigated first.
2. **Quality Range:** Average restaurant rating ranges from **2.40** (Hotel Zara Hi-Fi) to **4.88** (AB's - Absolute Barbecues). Top performers are barbecue, buffet and sit-down places.
3. **Price and Photos Matter:** Restaurants above the median cost average **3.78** stars vs **3.42** for cheaper ones, and reviews with pictures average **3.86** vs **3.53** without. Both differences are statistically significant (Mann-Whitney U, p < 0.01).
4. **Cuisine Effect:** Mediterranean (4.55), European (4.38) and American (3.93) are best rated, while Fast Food (3.32), Andhra (3.36) and Biryani (3.38) are lowest, even though they are high-demand cuisines.
5. **Text Is the Signal:** Numeric features correlate only weakly with rating (maximum 0.14), so an NLP model on the review text is needed. Weekend vs weekday has no significant effect on sentiment (chi-square p = 0.26).
6. **Six Restaurant Segments:** K-Means (k = 6, silhouette 0.287) separates star performers (cost about INR 1,340, rating 4.65) from low-price, low-quality outlets (rating 2.86, only 40% positive reviews).

### Model Performance (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression (tuned) - FINAL** | **0.8674** | **0.8966** | **0.8923** | **0.8945** | **0.9424** |
| Naive Bayes (tuned) | 0.8584 | 0.8958 | 0.8772 | 0.8864 | 0.9356 |
| Random Forest (tuned) | 0.8594 | 0.8817 | 0.8971 | 0.8893 | 0.9352 |

The final model was selected by the best cross-validated F1 (0.912) and is fully explainable: *awesome, best, love, amazing, delicious* push a review towards Positive, while *not, worst, bad, disappointing, tasteless* push it towards Negative.

---

## 🛠️ Tech Stack & Libraries Used

- **Language:** Python 3.10+
- **Data Manipulation:** pandas, numpy
- **Data Visualization:** matplotlib, seaborn, wordcloud
- **Statistics:** SciPy (Mann-Whitney U, chi-square)
- **NLP & Machine Learning:** scikit-learn (TF-IDF, Logistic Regression, Naive Bayes, Random Forest, K-Means, PCA, TruncatedSVD), NLTK (optional)
- **Model Persistence:** joblib
- **Environment:** Jupyter Notebook / Google Colab

---

## 🚀 How to Run the Project

1. Clone or extract the submission folder:

   ```bash
   cd Zomato_Project_Package
   ```

2. Install the dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scipy scikit-learn joblib nltk wordcloud
   ```

3. Launch Jupyter Notebook:

   ```bash
   jupyter notebook Zomato_Restaurant_ML_Capstone.ipynb
   ```

4. Run all cells sequentially (**Kernel → Restart & Run All**). The notebook finds the CSV files automatically in `data/` or next to the notebook, and regenerates every chart, statistic and the saved model.

5. To score new reviews with the saved model, see Section 8 of the notebook. In short:

   python
   import joblib
   model = joblib.load("model/zomato_sentiment_model.joblib")
   model.predict_proba(["food was delicious and staff were polite"])[:, 1]   # probability of a positive review
   ```

   The input text should be cleaned with the same steps used in training (see `clean_text` in the notebook).

> **Note:** NLTK data (stop words, WordNet) is downloaded automatically when internet access is available. If it is not, the notebook falls back to scikit-learn stop words and a rule-based lemmatizer, so it still runs without errors.

---

## ⚠️ Limitations & Future Work

- Only the 100 latest reviews per restaurant are available, so time trends are sample-driven.
- Sentiment labels come from star ratings (3 stars are treated as Not positive); sarcasm can mislead the model.
- Only one area of one city is covered.
- Future work: transformer models (BERT), aspect-based sentiment (food / service / price), a three-class model, a live prediction app (Streamlit / Flask) and scheduled retraining.
