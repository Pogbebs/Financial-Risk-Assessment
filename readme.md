# Predicting Credit Risk Ratings Using Machine Learning Models

## Project Overview
This project focuses on building machine learning models to predict the credit risk ratings of customers. The target variable is **Risk Rating**, which is categorized into three levels: *High (3)*, *Medium (2)*, and *Low (1)*. A variety of classification algorithms, including multinomial logistic regression, random forests, support vector machines, and ensemble methods, are employed to evaluate their predictive performance. 

---

## Dataset Description
The dataset consists of various demographic, financial, and behavioral attributes of customers. The main goal is to classify the risk associated with each customer into one of three categories based on the available features.

### Key Features:
- **Demographics**: Gender, Marital Status, Education Level, Age.
- **Financial Attributes**: Income, Debt-to-Income Ratio, Assets Value.
- **Behavioral Attributes**: Employment Status, Loan Purpose, Payment History, Previous Defaults.
- **Risk Rating**: Target variable with three levels: High, Medium, Low.

---

## Data Preprocessing

### 1. Missing Values Handling
- Missing values were identified and diagnosed using visualizations (e.g., boxplots) and statistical tests (e.g., t-tests).
- Significant relationships between missingness and other variables were observed and used for informed imputation.
- The **Random Forest** imputation method was employed to address missing values in the dataset.

### 2. Data Cleaning
- Categorical variables were encoded using custom mapping:
  - Gender (e.g., Male = 1, Female = 2, Non-binary = 3).
  - Marital Status, Education Level, Loan Purpose, Employment Status, and Payment History followed a similar mapping strategy.
- Columns with high cardinality or redundancy (e.g., City, State, Country) were removed.

### 3. Outlier Detection and Removal
- Outliers were identified using **Mahalanobis Distance** and visualized with chi-square Q-Q plots.
- Observed outliers were removed to enhance data quality.

---

## Data Visualization
Various visualizations were employed to understand data distribution and relationships:
- **Barplots**: Average income by gender and other categorical variables.
- **Boxplots**: Income by risk rating, stratified by gender, marital status, and employment.
- **Correlation Heatmap**: Depicting relationships between numeric features.

---

## Modeling Approach
### 1. Data Partitioning
- The dataset was split into **training (80%)** and **validation (20%)** sets.

### 2. Classification Models
A range of machine learning models was implemented, including:
- **Multinomial Logistic Regression**: Suitable for the multi-class nature of the target variable.
- **Random Forest**: A robust ensemble learning method that handles complex interactions well.
- **Support Vector Machine (SVM)**: Leveraging hyperplane separation for classification.
- **Tree-Based Models**: Visualized using `rpart` for interpretability.

### 3. Ensemble Methods
- **Bagging**: Built using bagged decision trees to reduce variance.
- **Boosting**: Implemented using **XGBoost** to improve weak learners.
- **Stacking**: Combined predictions from base models (tree, KNN, logistic regression) into a meta-model.

---

## Evaluation
Each model was evaluated using confusion matrices, capturing metrics such as:
- Accuracy
- Precision
- Recall
- F1-Score

### Key Insights:
- **Multinomial Logistic Regression**: Effective for interpreting multi-class relationships.
- **Random Forest**: Achieved competitive accuracy with strong generalization.
- **Boosting**: Delivered the best performance by refining weak learners.
- **Stacking**: Combined the strengths of base models for an improved predictive power.

---

## Insights and Recommendations
1. **Missing Values**:
   - Some attributes, such as **Assets Value**, showed a significant relationship with variables like marital status change and employment status. Addressing this missingness improves data integrity.
   
2. **Feature Importance**:
   - Attributes such as **Debt-to-Income Ratio**, **Payment History**, and **Employment Status** play a critical role in predicting credit risk.
   
3. **Model Performance**:
   - Ensemble methods like **Boosting** and **Stacking** provided the highest predictive accuracy, demonstrating their ability to capture complex patterns in the data.

---

## Conclusion
This project successfully predicts credit risk ratings using machine learning models. The combination of data preprocessing, robust modeling techniques, and careful evaluation ensures reliable predictions that can be leveraged for better risk management strategies in financial applications.

--- 

## Tools and Libraries
- **Programming Language**: R
- **Libraries**: `caret`, `ggplot2`, `neuralnet`, `e1071`, `adabag`, `missForest`, `nnet`, `rpart.plot`, `patchwork`.

---
