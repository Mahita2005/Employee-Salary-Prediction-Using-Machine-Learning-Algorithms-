# 💼 Employee Salary Prediction

A machine learning app to predict if an employee earns more than $50K/year using demographic and job-related features.

---

## 🚀 Models Used

| Model                  | Accuracy |
|------------------------|----------|
| Gradient Boosting      | **86.00%** |
| Random Forest          | 84.83%   |
| SVM                    | 84.19%   |
| KNN                    | 81.68%   |
| Logistic Regression    | 81.97%   |

✅ Best: **Gradient Boosting Classifier**

---

## 📋 Features

- Age, Workclass, Education, Marital Status  
- Occupation, Relationship, Race, Sex  
- Capital Gain/Loss, Hours per week  

Data is preprocessed using `LabelEncoder` and `MinMaxScaler`.

---

## 🖥️ Run the App

```bash

streamlit run app.py
