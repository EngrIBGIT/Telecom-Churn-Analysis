# End-to-End Customer Churn Prediction with Business Insights

## Project Overview

This project provides an end-to-end solution for customer churn prediction in the telecommunications industry. It encompasses data loading and cleaning, extensive exploratory data analysis (EDA), feature engineering, development and evaluation of various machine learning models, postdictive analysis of misclassifications, and actionable business recommendations, including a customer prioritization framework for targeted retention efforts.

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [1. Data Loading and Initial Inspection](#1-data-loading-and-initial-inspection)
  - [2. Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
  - [3. Feature Engineering](#3-feature-engineering)
  - [4. Data Preparation for Modeling](#4-data-preparation-for-modeling)
  - [5. Model Training and Evaluation](#5-model-training-and-evaluation)
  - [6. Postdictive Analysis](#6-postdictive-analysis)
- [Model Performance Summary](#model-performance-summary)
- [Best Performing Model](#best-performing-model)
- [Actionable Business Recommendations](#actionable-business-recommendations)
- [Customer Prioritization Framework](#customer-prioritization-framework)
- [Next Steps & Future Work](#next-steps--future-work)
- [Conclusion](#conclusion)

## Problem Statement

Customer churn is a critical challenge for telecom companies, directly impacting revenue and profitability. High churn rates necessitate costly customer acquisition efforts and reduce customer lifetime value (CLTV). The primary goal of this project is to build a predictive model that can accurately identify customers at high risk of churning, enabling proactive intervention strategies to retain valuable customers.

## Dataset

The project utilizes the `telecom_churn.csv` dataset, which contains various customer attributes and services. The `Churn` column is the target variable, indicating whether a customer has churned ('Yes' or 'No').

**Key Features:**
- `customerID`: Unique customer identifier.
- `gender`, `SeniorCitizen`, `Partner`, `Dependents`: Demographic and family information.
- `tenure`: Number of months the customer has been with the company.
- `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`: Service subscription details.
- `Contract`: Contract type (Month-to-month, One year, Two year).
- `PaperlessBilling`, `PaymentMethod`: Billing and payment preferences.
- `MonthlyCharges`: Current monthly charges.
- `TotalCharges`: Total charges incurred by the customer.

## Methodology

### 1. Data Loading and Initial Inspection

- The `telecom_churn.csv` dataset was loaded into a pandas DataFrame.
- Initial inspection (`df.head()`, `df.info()`, `df.describe()`) revealed 7043 entries and 21 columns.
- The `TotalCharges` column was identified as `object` type with 11 missing values, which required cleaning.
- The `Churn` column was confirmed as the target variable, indicating 'Yes' for churn and 'No' for no churn.

### 2. Data Cleaning and Preprocessing

- **`TotalCharges` Handling**: Converted to numeric (`float64`), coercing errors. Missing values were imputed using the median (`1397.475`).
- **`Churn` Target Variable**: Converted from categorical ('Yes', 'No') to binary numerical (1 for 'Yes', 0 for 'No').
- **Class Imbalance**: Noted a significant class imbalance with approximately 26.5% churned customers (1869 'Yes' vs. 5174 'No').

### 3. Exploratory Data Analysis (EDA)

Comprehensive EDA was performed to understand data characteristics, churn distribution, and relationships between features and churn.

**Key Findings from EDA:**

- **Tenure**: Customers with shorter tenure are significantly more likely to churn.
- **MonthlyCharges**: Higher monthly charges show a tendency for increased churn.
- **TotalCharges**: Lower total charges (often correlating with shorter tenure) are associated with higher churn rates.
- **Contract**: 'Month-to-month' contracts exhibit substantially higher churn rates compared to longer-term contracts.
- **InternetService**: 'Fiber optic' users have a higher churn rate, suggesting potential service dissatisfaction.
- **PaymentMethod**: 'Electronic check' users are more prone to churn.
- **OnlineSecurity & TechSupport**: Customers without these services tend to churn more, indicating their role as 'stickiness' factors.
- **SeniorCitizen**: Senior citizens have a higher churn rate.
- **Gender**: Appears to have little correlation with churn.
- **Correlation Matrix**: Confirmed strong relationships, e.g., negative correlation of `tenure` and `TotalCharges` with `Churn`, and positive correlation of `MonthlyCharges` and `SeniorCitizen` with `Churn`.

### 4. Feature Engineering

Several new features were engineered to capture more predictive insights:

- **`TenureGroup`**: Categorized `tenure` into 'New' (0-12 months), 'Medium' (13-48 months), and 'Long-term' (>48 months) to represent customer lifecycle stages.
- **`TotalServices`**: Count of services subscribed by each customer, indicating engagement and dependency.
- **`HasInternetSecurity`**: Binary indicator (1/0) for customers with any internet security-related services (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`).
- **`HasStreamingServices`**: Binary indicator (1/0) for customers with `StreamingTV` or `StreamingMovies`.
- **`MonthlyChargesPerTenure`**: `MonthlyCharges` divided by `tenure`, serving as a proxy for 'value for money' or early cost sensitivity.
- **`IsMonthToMonthContract`**: Binary indicator (1/0) for month-to-month contract holders.
- **`IsElectronicCheckPayment`**: Binary indicator (1/0) for customers using electronic checks for payment.

These features were analyzed for their correlation with Churn, confirming their relevance and predictive potential.

### 5. Data Preparation for Modeling

- **Feature-Target Split**: Features (`X`) and target variable (`y`) were separated, dropping `customerID` and `Churn` from `X`.
- **One-Hot Encoding**: All categorical features in `X` (including the newly engineered `TenureGroup`) were one-hot encoded using `pd.get_dummies` with `drop_first=True` to prevent multicollinearity.
- **Train-Test Split**: The data was split into training (80%) and testing (20%) sets using `train_test_split` with `random_state=42` and `stratify=y` to maintain class distribution.

### 6. Model Training and Evaluation

Five classification models were trained and evaluated using cross-validation and various metrics:

- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **XGBoost Classifier**
- **Gradient Boosting Classifier**

Each model underwent hyperparameter tuning using `GridSearchCV` with 5-fold cross-validation and `roc_auc` as the scoring metric (due to class imbalance). Evaluation included confusion matrices, classification reports (precision, recall, f1-score), ROC-AUC scores, and ROC curves.

### 7. Postdictive Analysis

A detailed analysis of model misclassifications (false positives and false negatives) was conducted using the best-performing Gradient Boosting model to understand its limitations and inform business strategies.

- **False Positives (Predicted Churn = 1, Actual Churn = 0)**: 85 customers. Characterized by short tenure, high monthly charges, month-to-month contracts, electronic checks, and fiber optic internet. The model flagged them as high-risk, but they did not churn. **Business Implication**: Wasted retention efforts.
- **False Negatives (Predicted Churn = 0, Actual Churn = 1)**: 194 customers. Characterized by moderate tenure, month-to-month contracts (though some longer-term), diverse internet services, and sometimes 'sticky' services. The model predicted they would stay, but they churned. **Business Implication**: Missed retention opportunities and direct revenue loss.

## Model Performance Summary

| Model                 | ROC-AUC | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
|-----------------------|---------|---------------------|------------------|--------------------|
| Logistic Regression   | 0.8417  | 0.6520              | 0.5160           | 0.5761             |
| Decision Tree         | 0.8294  | 0.6409              | 0.4439           | 0.5245             |
| Random Forest         | 0.8444  | 0.6735              | 0.4412           | 0.5331             |
| XGBoost               | 0.8452  | 0.6601              | 0.5134           | 0.5776             |
| Gradient Boosting     | 0.8458  | 0.6792              | 0.4813           | 0.5634             |

*(Note: Metrics for Class 1 (Churn) are provided)*

## Best Performing Model

Based on the evaluation metrics, the **Gradient Boosting Classifier** was selected as the best-performing model. It achieved the highest ROC-AUC score of **0.8458**, indicating its superior ability to distinguish between churned and non-churned customers across various thresholds. While other models had comparable F1-scores or slightly higher recall, Gradient Boosting provided the best overall balance between identifying true churners and minimizing false alarms, which is crucial for resource-constrained retention efforts.

## Actionable Business Recommendations

1.  **Targeted Retention for Month-to-Month Contract Holders**: Implement aggressive, personalized incentive programs (discounts, premium upgrades) to encourage conversion to longer-term contracts, addressing the highest churn risk factor.
2.  **Enhance Early-Life Customer Engagement & Onboarding**: Develop proactive onboarding journeys for new customers (first 12 months) with personalized check-ins, dedicated support, and early satisfaction surveys to mitigate churn during the critical initial phase.
3.  **Improve Reliability and Perceived Value of Fiber Optic Internet**: Conduct targeted satisfaction audits and service optimization for Fiber Optic users to address unmet expectations and enhance perceived value, as these customers show higher churn.
4.  **Incentivize Secure and Automated Payment Methods**: Offer small incentives (e.g., bill credit) to encourage customers to switch from electronic checks to more automated payment methods, as electronic check users have higher churn rates.
5.  **Proactive Support for Senior Citizens**: Implement dedicated support channels and simplified communication for senior citizens to address their unique needs and enhance loyalty.
6.  **Bundle and Promote Stickiness-Enhancing Services**: Create attractive service bundles integrating Internet Security, Online Backup, Device Protection, and Tech Support to increase customer dependency and trust.
7.  **Re-evaluate Pricing Strategies for High Monthly Charge Customers**: Conduct competitive pricing analysis and offer discounted rates or value-added services to high-charge customers, especially those not on long-term contracts, to address perceived cost issues.
8.  **Personalized Engagement for Loyal (Long-Term) Customers**: Implement loyalty programs with exclusive benefits, proactive check-ins, and early access to new features to maintain satisfaction among high-CLTV customers.
9.  **Targeted Interventions for Customers with Low Total Services**: Identify customers with few services and offer compelling cross-selling promotions to increase their engagement and dependency on the provider.
10. **Analyze Misclassified Customers for Deeper Insights**: Conduct qualitative research (surveys, interviews) with samples of False Positive and False Negative customers to uncover latent churn drivers and uncaptured loyalty factors, further refining retention strategies.

## Customer Prioritization Framework

A framework for prioritizing high-risk customers for intervention was developed, segmenting customers into tiers based on their predicted churn probability (High-Risk vs. Low-Risk) and customer value (using `TotalCharges` as a proxy: Low-Value, Medium-Value, High-Value). This ensures retention efforts are focused on the most impactful customers.

**Example Tiers:**
- **High-Risk-High-Value (HRHV)**: Top priority; immediate, personalized, comprehensive interventions (e.g., dedicated account managers, exclusive offers).
- **High-Risk-Medium-Value (HRMV) / High-Risk-Low-Value (HRLV)**: Targeted interventions ranging from significant attention to more automated efforts.
- **Low-Risk-High-Value (LRHV)**: Focus on loyalty programs, upselling/cross-selling, and continued satisfaction.
- **Low-Risk-Medium-Value (LRMV) / Low-Risk-Low-Value (LRLV)**: Minimal efforts, focusing on broad loyalty programs and potential value growth.

## Next Steps & Future Work

-   **Deployment**: Integrate the Gradient Boosting model into the company's CRM or customer analytics platform to generate real-time churn predictions.
-   **A/B Testing of Interventions**: Systematically A/B test the proposed business recommendations to measure their actual impact on churn reduction and ROI.
-   **Model Monitoring & Retraining**: Continuously monitor model performance in production and retrain with fresh data to adapt to changing customer behaviors and market conditions.
-   **Feature Expansion**: Explore additional features such as customer service interaction history, website/app usage data, and competitive landscape information.
-   **Threshold Optimization**: Experiment with different classification thresholds for the Gradient Boosting model to balance false positives and false negatives based on business costs and benefits.
-   **Power BI Dashboard Implementation**: Fully implement the Power BI dashboard as designed to provide an interactive and dynamic view of churn metrics and customer segments.

## Conclusion

This project provides a robust, end-to-end customer churn prediction solution tailored for the telecom industry. By combining advanced machine learning with practical business insights, the company can proactively identify and retain valuable customers, optimize resource allocation, and enhance customer loyalty, ultimately safeguarding revenue and strengthening its competitive position.