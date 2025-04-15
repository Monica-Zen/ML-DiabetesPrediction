# 🧊 SugarSage

A wise guide for tracking and predicting diabetes.  
Suggesting proactive and futuristic health management.

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://app-starter-kit.streamlit.app/)

## GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/Monica-Zen/ML-DiabetesPrediction)

---

## Project Overview

**SugarSage** is a machine learning-powered application aimed at predicting diabetes risk based on key patient attributes. Its focus is to provide insights to individuals and healthcare professionals, fostering proactive and informed decision-making.

---

## Dataset Description

The dataset contains the following attributes:

- **Age**: Represents the patient's age in years. Age is a significant risk factor for diabetes, as risk increases with age.
- **Gender**: Indicates the patient's gender, which may influence diabetes risk.
- **Body Mass Index (BMI)**: A measure based on height and weight to classify weight categories. Higher BMI correlates with higher diabetes risk.
- **Chol (Cholesterol)**: Total cholesterol level in the blood. High cholesterol is a risk factor for diabetes and heart disease.
- **TG (Triglycerides)**: Blood triglyceride levels. Elevated levels increase risk of heart disease and diabetes.
- **HDL (High-Density Lipoprotein)**: "Good" cholesterol that removes excess cholesterol. Higher HDL levels are beneficial.
- **LDL (Low-Density Lipoprotein)**: "Bad" cholesterol that contributes to artery plaque buildup, increasing diabetes risk.
- **Cr (Creatinine)**: Indicates kidney function. Kidney disease may be linked to diabetes risk.
- **BUN (Blood Urea Nitrogen)**: Reflects kidney and liver function. Abnormal levels may signal disorders related to diabetes.
- **Diagnosis**: Indicates whether a patient has diabetes.

---

## Objectives

- Predict diabetes risk based on demographic and health data.
- Analyze the impact of each attribute on diabetes prediction.
- Offer a user-friendly interface for exploring prediction results.

---

## Features

### 1. **Interactive Dashboard**

- Explore predictions with visualizations.
- View data distributions, scatter plots, and correlation heatmaps.

### 2. **Hyperparameter Tuning**

- Adjust hyperparameters for models like Logistic Regression, Decision Tree, Random Forest, SVM, and K-Nearest Neighbor to optimize performance.

### 3. **Feature Importance**

- Visualize the importance of each feature in the prediction process using horizontal bar charts.

### 4. **Model Evaluation**

- Evaluate models with metrics like Accuracy, Precision, Recall, and F1 Score displayed as pie charts.
- View prediction probabilities with a dynamic progress bar.

### 5. **Data Preprocessing**

- Handle missing values, normalize data, and convert categorical variables into numerical format.
- Split the dataset into training and testing sets.

### 6. **Data Visualizations**

- **Age Distribution**: View the distribution of patient ages.
- **Gender Count**: Analyze the gender distribution in the dataset.
- **BMI vs Cholesterol**: Explore the relationship between BMI and cholesterol levels using scatter plots.
- **Correlation Heatmap**: Visualize correlations between features using a heatmap.
- **Diagnosis Count**: View the distribution of diabetes diagnoses.

---

## Models Supported

- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbor (KNN)**

Each model includes hyperparameter tuning options to optimize performance.

---

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/Monica-Zen/ML-DiabetesPrediction.git
   cd ML-DiabetesPrediction
   ```
