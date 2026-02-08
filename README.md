# Hybrid Collaborative Filtering with KNN Recommenders

> Design, optimization, and evaluation of user-based, item-based, and hybrid neighborhood collaborative filtering models under sparse rating data.

---

## 📌 Overview

Recommender systems are a core component of modern digital platforms, enabling personalized content delivery in environments characterized by sparse and noisy user feedback.

This project presents an end-to-end study of **neighborhood-based collaborative filtering (CF)** methods, progressing from simple statistical baselines to optimized **User-KNN**, **Item-KNN**, and a **hybrid ensemble model**. The work emphasizes model interpretability, hyperparameter optimization, and empirical performance evaluation using MAE and RMSE.

The project originated as academic coursework and was later **refactored into a modular, production-style ML codebase** suitable for portfolio presentation.

---

## 🎯 Problem Statement

Given a sparse user–item rating matrix:

- Users interact with only a small subset of items
- Similarity estimates are noisy under sparsity
- Single-model approaches often exhibit high bias or variance

The goal is to predict missing ratings accurately while balancing personalization and robustness.

This project investigates:
1. How different neighborhood-based CF strategies perform under identical conditions
2. The impact of similarity metrics and neighborhood size
3. Whether hybridization improves predictive accuracy

---

## 🧠 Models Implemented

### 1. Baseline Models
- **User Average** – per-user mean rating
- **Item Average** – per-item mean rating

Used as lower-bound performance benchmarks.

---

### 2. User-Based KNN Collaborative Filtering
- Similarity metrics: **Cosine**, **Pearson (mean-centered)**
- Tuned hyperparameters:
  - Neighborhood size (`k`)
  - Similarity exponent
  - Similarity threshold
- Fallback strategy:
  - User mean → global mean

**Observation:** User-KNN struggles under sparse user overlap despite optimization.

---

### 3. Item-Based KNN Collaborative Filtering
- Similarity computed between item vectors
- More stable neighborhoods due to higher rating density
- Evaluated across multiple `k` values and similarity metrics

**Observation:** Item-KNN consistently outperforms User-KNN in sparse settings.

---

### 4. Hybrid Ensemble Model ⭐

A weighted combination of User-KNN and Item-KNN predictions:
```text
Hybrid prediction:

R̂(u,i) = λ · R̂_user(u,i) + (1 − λ) · R̂_item(u,i)
```

- λ tuned empirically
- Best performance achieved at **λ ≈ 0.40**
- Demonstrates complementary strengths of both models

---

## 📊 Evaluation Metrics

Models are evaluated using:

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**

Only observed test ratings are considered (missing entries ignored).

---

## 🧪 Experimental Results

| Model | MAE | RMSE |
|------|-----|------|
| User Average Baseline | 0.751 | 0.954 |
| Item Average Baseline | 0.751 | 0.954 |
| Best User-KNN (Pearson, k=20) | 0.781 | 0.983 |
| Best Item-KNN (Pearson, k=15) | 0.751 | 0.954 |
| **Hybrid Model (λ = 0.40)** | **0.730** | **0.923** |

📉 The hybrid model reduces RMSE by ~3% compared to the best single-model approach.

---

## 📈 Visual Results

All performance plots are generated in the experiment notebook and saved under:

```text
reports/figures/
```


Including:
- Item-KNN performance vs neighborhood size
- User-KNN performance vs neighborhood size
- Hybrid MAE/RMSE vs λ

---

## 🧩 Project Structure

```text
recommender-systems-knn-hybrid/
│
├── notebooks/
│ └── recommender_experiments.ipynb # Experiments, plots, analysis
│
├── src/
│ ├── metrics.py # MAE, RMSE evaluation
│ ├── similarity.py # Cosine & Pearson similarity
│ ├── knn_models.py # Baselines, User-KNN, Item-KNN, grid search
│ └── hybrid.py # Hybrid ensemble & lambda tuning
│
├── reports/
│ ├── figures/ # Saved plots
│ └── performance_summary.pdf # Technical report
│
├── README.md
```


**Design principle:**
- `src/` contains reusable, production-style model code
- `notebooks/` orchestrate experiments and visualization only

---

## 🔬 Key Insights

- Item-based collaborative filtering is more robust under sparse data
- Pearson similarity outperforms cosine similarity
- Moderate neighborhood sizes (`k ≈ 10–15`) provide the best bias–variance trade-off
- Hybrid ensembling improves accuracy by combining complementary signals

---

## ⚠️ Limitations

- Cold-start problem not addressed
- Quadratic similarity computation limits scalability
- Explicit feedback only (no implicit interactions)

---

## 🚀 Future Work

- Matrix factorization (SVD, ALS)
- Approximate nearest neighbor search
- Implicit feedback modeling
- Deep learning–based recommender systems

---

## 📚 References

- Ricci et al., *Recommender Systems Handbook*, Springer
- Sarwar et al., *Item-Based Collaborative Filtering Recommendation Algorithms*, WWW
- Netflix Prize documentation
- Industry recommender system design practices

---

## 👤 Author

**Hishikesh Phukan**  
Master of Data Science  
Melbourne, Australia

---

⭐ This project demonstrates end-to-end recommender system design, from algorithmic fundamentals to empirical evaluation and hybrid model engineering.
