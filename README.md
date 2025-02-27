# 🩺 Cancer Risk Prediction Using Machine Learning  

## 📌 Project Overview  
This project aims to predict **cancer risk** using **patient data** and multiple machine learning models. 
The dataset consists of **1500 patients aged 20-80** with **8 predictors**, including lifestyle factors such as smoking, alcohol intake, and physical activity.  

I implemented and compared different **classification models** to determine the most effective method for risk prediction.  

### 🔍 Key Insights  
- **Logistic Regression, Linear Discriminant Analysis (LDA), and Quadratic Discriminant Analysis (QDA)** all performed similarly.  
- **K-Nearest Neighbors (KNN)** produced slightly **higher training accuracy** but had a **decreased recall score**.  
- **Precision, Recall, and F1-score** were analyzed due to the **imbalanced dataset** (557 cancer cases vs. 943 non-cancer cases).  
- **ROC curves** helped assess the **stability** and **predictive power** of the models.  

---

## 📂 Dataset  
- **Source**: Kaggle – Cancer Prediction Dataset  
- **Data Composition**:  
  - **Binary classification**: `0` (No Cancer) / `1` (Cancer)  
  - **1500 patient records**  
  - **8 predictors**: Age, Gender, BMI, Smoking, Genetic Risk, Physical Activity, Alcohol Intake, Cancer History  

🗂 **Files in this repository:**  
- `ML Cancer project.ipynb` → Jupyter Notebook main code 
- `cancerdata.csv` → Raw dataset used in the study before preprocessing
- `ML cancer prediction PPT (1).pdf` → Project slides summarizing the findings & showing graphs

---

## 📊 Methodology  
1. **Data Preprocessing**:  
   - Handled missing values and encoded categorical variables.  
   - Standardized numerical predictors to improve model performance.  
   - Addressed class imbalance through **performance metric selection** rather than resampling.  

2. **Exploratory Data Analysis (EDA)**:  
   - **Correlation Heatmap** to check relationships between variables.  
   - **Feature Distribution Plots** to analyze the dataset.  

3. **Model Selection & Training**:  
   - **Logistic Regression**: Baseline binary classification model.  
   - **Linear Discriminant Analysis (LDA)**: Assumes normally distributed data with a common covariance matrix.  
   - **Quadratic Discriminant Analysis (QDA)**: Similar to LDA but allows different covariance matrices per class.  
   - **K-Nearest Neighbors (KNN)**: Non-parametric model using distance-based classification.  

4. **Evaluation Metrics**:  
   - **Accuracy** (not always reliable for imbalanced data).  
   - **Precision & Recall** (to evaluate false positives & false negatives).  
   - **F1-score** (balance between precision & recall).  
   - **ROC-AUC Score** (overall performance assessment).  

---

## 📈 Results & Findings  
✔ **Logistic Regression & LDA performed identically** in terms of accuracy and AUC-ROC.  
✔ **QDA produced slightly different results** but had a higher test error rate.  
✔ **KNN showed the highest training accuracy**, but its **recall and F1-score dropped**.  
✔ **ROC curves** showed that all models had decent classification power, with AUC values above 0.85.  

📊 **Example Performance Summary:**  
| Model  | Training Error | Test Error | AUC-ROC | Accuracy | Precision (Cancer) | Recall (Cancer) |  
|--------|---------------|------------|---------|----------|------------------|---------------|  
| Logistic Regression | 0.153 | 0.140 | 0.94 | 86% | 0.87 | 0.75 |  
| LDA | 0.153 | 0.140 | 0.94 | 86% | 0.87 | 0.75 |  
| QDA | 0.153 | 0.147 | 0.85 | 85% | 0.86 | 0.74 |  
| KNN (K=5) | 0.113 | 0.190 | 0.89 | 81% | 0.82 | 0.65 |  

---


## 📌 Future Improvements  
- **Optimize feature selection** using advanced techniques to reduce model complexity.  
- **Test additional models** such as **Random Forest** or **Neural Networks** for better performance.  
- **Address class imbalance** by trying **resampling methods** or **cost-sensitive learning**.  
- **Incorporate clinical validation** by comparing predictions with real-world patient data.  

---

## 📜 References  
1. Kaggle – [Cancer Prediction Dataset](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset).  
2. Scikit-Learn – [Machine Learning Documentation](https://scikit-learn.org/stable/documentation.html).  
3. Google Developers – [Precision & Recall Metrics](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall).  
4. James, G., Witten, D., Hastie, T., Tibshirani, R. – *An Introduction to Statistical Learning*, 2023.  

