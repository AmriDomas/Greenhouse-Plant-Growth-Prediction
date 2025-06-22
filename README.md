# 🌿 Greenhouse Plant Growth Prediction using XGBoost

This project applies machine learning to predict plant growth stages in a greenhouse environment using environmental sensor data. It leverages **XGBoost** for multiclass classification and includes explainability techniques to better understand model behavior.

## 📁 Project Overview

- **Objective**: Predict plant growth stages.
- **Model**: XGBoost Classifier (multiclass)
- **Evaluation**: Macro-average ROC AUC
- **Interpretation**: SHAP values and feature importance visualization

---

## 🔍 Dataset

The dataset includes time-series environmental data recorded inside a greenhouse, along with plant growth labels:

| Feature | Description |
|--------|-------------|
| ACHP | Average chlorophyll content per plant, an indicator of photosynthetic activity |
| PHR | Plant height rate – measures the vertical growth over time. |
| AWWGV | Average wet weight of vegetative growth – total fresh weight of the above-ground parts. |
| ALAP | Average leaf area per plant – surface area of leaves which impacts photosynthesis. |
| ANPL | Average number of leaves per plant – indicates plant maturity and foliage density. |
| ARD | Average root diameter – thickness of roots, relevant to nutrient uptake. |
| ADWR | Average dry weight of roots – total root biomass after drying. |
| PDMVG | Percentage of dry matter in vegetative growth – measures solid content in shoots. |
| ARL | Average root length – indicates root development and depth. |
| AWWR | Average wet weight of roots – total fresh root weight. |
| ADWV | Average dry weight of vegetative parts – dried mass of above-ground plant parts. |
| PDMRG | Percentage of dry matter in root growth – solid content of the root system. |
| Class | Categorical label indicating the experimental group: SA, SB, SC (Traditional Greenhouse), TA, TB, TC (IoT-based Greenhouse). |

---

## 🚀 Model Training

- **Algorithm**: XGBoost (`XGBClassifier`)
- **Data Split**: Train/Test (e.g., 80/20)
- **Target Type**: Multiclass

```python
from xgboost import XGBClassifier
model = XGBClassifier(objective='multi:softprob', num_class=6)
model.fit(X_train, y_train)
```

## 📊 Evaluation Metrics
 - Accuracy
 - Confusion Matrix
 - Macro ROC AUC Score
 - Classification Report
   ```python
   from sklearn.metrics import roc_auc_score
   roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr', average='macro')
   ```

## 🧠 Model Explainability

 - Feature Importance: Plotting XGBoost built-in importance
 - SHAP Values: Interpreting global and local predictions
   ```python
   import shap
   explainer = shap.Explainer(model)
   shap_values = explainer(X_test)
   shap.summary_plot(shap_values, X_test)
   ```

## 📈 Results Summary

 - Top Features: PHR, ARD and ACHP levels are most influential
 - ROC AUC: Achieved macro AUC of ~1 on test set
 - SHAP Insights: Identified feature thresholds affecting class changes

## 📦 Dependencies

 - Python 3.8+
 - xgboost
 - shap
 - pandas
 - scikit-learn
 - matplotlib / seaborn
   ```python
   pip install -r requirements.txt
   ```

## 📁 Folder Structure

```kotlin
├── Greenhouse_Plant_Growth_Prediction.ipynb
├── data/
│   └── Greenhouse Plant Growth Metrics.csv
├── models/
│   └── xgb_model.pkl
├── plots/
│   └── shap_summary.png
├── README.md
└── requirements.txt
```

## ✅ Use Cases

 - Greenhouse automation and monitoring
 - Smart agriculture decision-making
 - Precision farming with real-time predictions

## 🤝 Contributing
Feel free to fork the repo, open issues, or submit pull requests to improve model performance or integrate real-time data sources.

## 📬 Contact

For questions or collaboration:
Amri Sidiq

📧 [amrisidiq@gmail.com]

🔗 [Linkedin](http://linkedin.com/in/muh-amri-sidiq)



