# Same Data, Different Models

This project demonstrates how different machine learning models behave when trained on the same dataset.

The goal is not to chase high accuracy, but to understand:
- Why some models fail
- Why some models overfit
- Why some models generalize better


## Project Goal

To compare multiple regression models using:
- The same dataset
- The same features
- The same train/test split

And observe key ML concepts:
- Bias vs Variance
- Underfitting vs Overfitting
- How well the model generalizes to new data


## Dataset

- California Housing Dataset
- Target variable: `median_house_value`
- Categorical column removed
- Missing values dropped
- Features scaled using StandardScaler


## Models Used

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor


## Evaluation Metrics

The models are evaluated using:
- R² Score (Train & Test)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Scatter plots are used to visualize:
- Actual values vs Predicted values

## Project Structure

```
same-data-different-models/
├── data/
│   └── housing.csv
├── src/
│   ├── linear_regression.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   └── main.py
├── result/
│   └── metrics.txt
├── requirements.txt
└── README.md
```


---
   ```bash
   git clone https://github.com/Sabirhusseinbalal/same-data-different-models.git
   ```
---

## Machine Learning (Beginner → Advanced)

---

### Stage 1: Regression Foundations
- [***simple-regression-lab***](https://github.com/sabirhusseinbalal/simple-regression-lab)
- [***house-price-prediction-ml***](https://github.com/sabirhusseinbalal/house-price-prediction-ml)
- 👉 **same-data-different-models**

---

### Stage 2: Regression Deep Dive
- [***regression-error-analysis***](https://github.com/sabirhusseinbalal/regression-error-analysis)
- feature-engineering-regression
- regression-from-scratch

---

### Stage 3: Classification Core
- binary-classification-basics
- credit-risk-classification
- threshold-tuning-classification

---

### Stage 4: Classification Depth
- class-imbalance-handling
- logistic-regression-from-scratch
- model-interpretability

---

### Stage 5: Unsupervised Learning
- customer-segmentation-clustering
- dimensionality-reduction
- clustering-comparison

---

### Stage 6: Association & Anomaly Detection
- market-basket-analysis
- anomaly-detection-dbscan
- anomaly-detection-isolation-forest

---

### Stage 7: Ensemble & Optimization
- ensemble-learning-ml
- hyperparameter-tuning

---

### Stage 8: Real-World ML Projects
- churn-prediction-system
- fraud-detection-system
- sales-forecasting-system

---







