# eNPS-prediction
Bachelor’s Thesis project focused on predicting individual eNPS responses using Machine Learning and Deep Learning models (MLP and LSTM), including data preprocessing, temporal feature engineering, hyperparameter optimization, and model evaluation on real workplace climate data.

# Predicting Individual eNPS Responses using Machine Learning and Deep Learning

## 📌 Project Overview

This repository contains the development of my Bachelor’s Thesis, focused on predicting the individual score that each employee will give to the eNPS (Employee Net Promoter Score) question — a key metric used to measure employee satisfaction and engagement within organizations.

The objective of this project is to build predictive models capable of estimating future employee eNPS responses using historical workplace climate data.

---

## 🎯 Objectives

- Predict individual employee responses to the eNPS question (scale 1–10)  
- Compare different Machine Learning and Deep Learning approaches  
- Evaluate model performance using regression metrics  
- Identify the best candidate model for potential real-world deployment  

---

## 🧠 Methodology

The project follows a complete end-to-end data science pipeline:

### Data Preparation
- Data cleaning and transformation  
- Removal of duplicates and missing values  
- Filtering employees with insufficient historical data  
- Feature engineering  
- Temporal window construction for sequential models  

### Data Processing
- Min-Max normalization  
- Dataset split by employee ID  
  - Training: 70%  
  - Validation: 15%  
  - Test: 15%  

### Model Development

The following model families were developed and compared:

- **MLP (Multilayer Perceptron) – Independent observations**
- **MLP with sequential information (flattened sequences)**
- **LSTM (Long Short-Term Memory) – Temporal sequence modeling**

### Hyperparameter Optimization

- Random search over architecture parameters  
- Optimization of:
  - Number of layers  
  - Number of neurons / units  
  - Dropout rates  
  - Dense layer size  

---

## 📊 Evaluation Metrics

Models were evaluated using standard regression metrics:

- Mean Squared Error (MSE)  
- Mean Absolute Error (MAE)  
- R² Score  

Additional evaluation:
- Learning curves  
- Real vs Predicted scatter plots  

---

## 🏆 Results Summary

The optimized LSTM model achieved the best overall performance, demonstrating the importance of temporal information in predicting employee satisfaction.

**Key takeaway:**  
Temporal modeling significantly improves predictive performance compared to non-sequential approaches.

---

## 🗂 Data Description

The dataset includes anonymized employee workplace experience data provided by **Happyforce**, including:

- Employee feedback scores related to workplace environment  
- Happiness Index (daily emotional self-perception metric)  
- Historical eNPS responses  

Sensitive attributes such as gender or birth date were intentionally excluded to preserve employee privacy.

---

## 🛠 Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  

---

## 🔒 Data Privacy

All data used in this project was anonymized.  
The project follows privacy-preserving practices and does not include any personally identifiable employee information.

---

## 🎓 Academic Context

This project was developed as part of my **Bachelor’s Thesis (Trabajo de Fin de Grado)**.

---

## 📬 Contact

If you have questions about the project, feel free to reach out.
