#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries and Modules

# In[109]:


# Basic Libraries
import pandas as pd
import numpy as np
import sys
import warnings
import time

# Ignore Warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Display Settings
pd.set_option('display.max_columns', None)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from IPython.display import display, Markdown
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Statistical Analysis
import scipy.stats as stats
from scipy.stats import chi2_contingency, mannwhitneyu

# Preprocessing & Encoding
from sklearn.experimental import enable_iterative_imputer  # Needed before using IterativeImputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, RobustScaler, FunctionTransformer
)
from sklearn.compose import ColumnTransformer
import category_encoders as ce
from sklearn.pipeline import Pipeline

# Feature Selection
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, SelectFromModel, f_classif
)


# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier

# Model Selection & Evaluation
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, cross_validate,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    fbeta_score, roc_curve, roc_auc_score, confusion_matrix,
    classification_report, make_scorer
)


# Imbalanced Learning
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler


# Hyperparameter Tuning Support
from scipy.stats import randint, uniform

#Save Model
import pickle

#Import streamlit
import streamlit as st


# # Predicting Customer Churn: A Data-Driven Strategy to Maximize Retention in E-Commerce
# ![cover.png](attachment:78b87e04-249a-4c22-b82a-046daf718705.png)
# **Created by : Meriani Alexandra & Nadame Kristina** 

# ## Business Understanding
# #### Introduction
# In e-commerce, the cost of acquiring a new customer is 5 to 25 times higher than retaining an existing one. Yet many businesses continue to spend heavily on blanket marketing campaigns that fail to identify and prioritize the customers most at risk of leaving.
# According to the dataset, approximately **1 in 6 customers (16.8%) churn**. If left unaddressed, this could lead to significant revenue loss, decreased customer lifetime value, and wasted marketing budgets. Simply offering discounts or loyalty rewards to all customers isn't efficient, it risks **misallocating resources** toward users who were never likely to leave in the first place. 
# 
# To address this issue effectively, it's crucial to first understand what customer churn actually means and how it's represented in the data. Customer churn refers to when a customer stops using a company’s product or service within a certain period of time. In this dataset, churn is represented by the label `1`, indicating that the **customer has churned**, while `0` means the **customer has remained active**.
# 
# **Resources** :
# * [Customer Retention Versus Customer Acquisition](https://www.forbes.com/councils/forbesbusinesscouncil/2022/12/12/customer-retention-versus-customer-acquisition)
# * [Zero Defections: Quality Comes to Services](https://hbr.org/1990/09/zero-defections-quality-comes-to-services)
# 
# #### Problem Statement 
# 1. What are the key factors that influence customer churn in our e-commerce platform?
# 2. How can we accurately predict which customers are at risk of churning, so that marketing efforts and retention budgets can be precisely targeted and minimizing wasted spend?
# 
# #### Goals
# 1. To identify the most significant factors that contribute to customer churn on the platform by analyzing patterns and behaviors in the dataset.
# 2. To develop a predictive model that can accurately classify customers as likely to churn or not, using historical data.
# 
# #### Analytical Approach
# Multiple classification models will be trained to predict churn. The models will be evaluated using F2-score, which prioritizes recall to identify customers who are most likely to churn. Feature importance will guide strategic recommendations for targeted retention efforts.
# 
# #### Metric Evaluation
# |                       |                                 Predicted: Not Churn (0)                                 | Predicted: Churn (1)                                                            |
# |-----------------------|:----------------------------------------------------------------------------------------:|---------------------------------------------------------------------------------|
# | Actual: Not Churn (0) | **True Negative (TN)**  Model predicts the customer will stay, and they do              | **False Positive (FP)**  Model predicts churn, but the customer actually stays |
# | Actual: Churn (1)     | **False Negative (FN)**  Model predicts the customer will stay, but they actually churn | **True Positive (TP)** Model correctly predicts churn                          |
# 
# **Type 1 Error: False Positive (FP)**
# - Consequence: Retention budget is wasted on customers who were never at risk of churning. This leads to inefficient spending of marketing resources.
# 
# **Type 2 Error: False Negative (FN)**
# - Consequence: High-risk customers are not identified and receive no intervention. This can result in customer loss and decreased revenue.
# 
# In this project, false negative are considered the most dangerous. Given these consequences, the model should focus on minimizing false negative. In other words, it’s more important to correctly identify customers who are likely to churn, even at the risk of a few false positive. Therefore, the primary metric for model evaluation is the F2-Score, which gives more weight to recall than precision.
# 
# #### Stakeholders
# **Customer Marketing Team**
# 
# The Customer Marketing Team is the key stakeholder in this project, as they are directly responsible for developing and implementing strategic marketing plans to drive customer engagement and loyalty. This project empowers them to identify and target high-risk customers more precisely, reducing wasted marketing spend and improving the effectiveness of retention efforts.

# In[2]:


df = pd.read_csv("E_Comm.csv")
df


# ## Data Understanding
# Data source : [Ecommerce Customer Churn Analysis and Prediction](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)
# #### Attribute Information
# | **Attribute**                   | **Data Type** | **Description**                                                                |
# |---------------------------------|---------------|--------------------------------------------------------------------------------|
# | **CustomerID**                  | int64         | Unique identifier for each customer                                            |
# | **Churn**                       | int64         | Indicates whether the customer has churned (1) or not (0)                      |
# | **Tenure**                      | float64       | Duration (in months) the customer has been with the company           |
# | **PreferredLoginDevice**        | object        | Device most frequently used by the customer to log in                          |
# | **CityTier**                    | int64         | Tier classification of the customer's city                                     |
# | **WarehouseToHome**             | float64       | Distance between the warehouse and the customer's home                         |
# | **PreferredPaymentMode**        | object        | Customer's most preferred method of payment                                    |
# | **Gender**                      | object        | Gender of the customer                                                         |
# | **HourSpendOnApp**              | float64       | Number of hours the customer spends on the mobile app                          |
# | **NumberOfDeviceRegistered**    | int64         | Total number of devices registered by the customer                             |
# | **PreferedOrderCat**            | object        | Most frequently ordered product category                                       |
# | **SatisfactionScore**           | int64         | Customer satisfaction score based on feedback                                  |
# | **MaritalStatus**               | object        | Marital status of the customer                                                 |
# | **NumberOfAddress**             | int64         | Total number of addresses added by the customer                                |
# | **Complain**                    | int64         | Indicates whether the customer raised a complaint last month (1 = Yes, 0 = No) |
# | **OrderAmountHikeFromlastYear** | float64       | Percentage increase in order amount compared to last year                      |
# | **CouponUsed**                  | float64       | Number of coupons used in the last month                                       |
# | **OrderCount**                  | float64       | Total number of orders placed in the last month                                |
# | **DaySinceLastOrder**           | float64       | Number of days since the last order was placed                                 |
# | **CashbackAmount**              | float64       | Average cashback received by the customer in last month                        |

# In[3]:


df.dtypes


# Next, the columns in the DataFrame are grouped by similar types of information. This helps make the data easier to read and understand, and also makes the analysis process more efficient.
# 
# 1. Customer Demographics
#    These describe who the customer is : `CustomerID`, `CityTier`, `WarehouseToHome`, `Gender`, `NumberOfDeviceRegistered`, `MaritalStatus`,       `NumberOfAddress`.
# 2. Customer Behaviour/ App usage
#    These describe how the customer interacts with the app : `PreferredLoginDevice`, `PreferredPaymentMode`, `PreferedOrderCat`, `HourSpendOnApp`, `Tenure`.
# 3. Customer Purchase History
#    These reflect purchase patterns and order history : `OrderAmountHikeFromlastYear`, `CouponUsed`, `OrderCount`, `DaySinceLastOrder`, `CashbackAmount`.
# 4. Customer Feedback/ Satisfaction
#    These reflect the customer’s opinion, experience, and level of satisfaction with the service or product : `SatisfactionScore`, `Complain`.
# 5. Target Variable
#    This reflect hether a customer has stopped using the service or remained active : `Churn`

# #### Univariate Distribution Check for Quantitative Features

# In[4]:


quantitative_features = [
    'Tenure', 'WarehouseToHome', 'HourSpendOnApp',
    'OrderAmountHikeFromlastYear', 'CashbackAmount',
    'NumberOfDeviceRegistered', 'NumberOfAddress',
    'CouponUsed', 'OrderCount', 'DaySinceLastOrder'
]

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(14, 20))
fig.suptitle('Univariate Distribution of Quantitative Features', fontsize=13, fontweight='bold', y=1.0)
axes = axes.flatten()  

for i, col in enumerate(quantitative_features):
    sns.histplot(df[col], kde=True, ax=axes[i], color= '#247c6d')
    axes[i].set_title(f'Histogram of {col}')  
    axes[i].set_xlabel(col)  
    axes[i].set_ylabel("Count")  

plt.tight_layout()  
plt.show()  


# Most of the visual distributions observed in the dataset are **not normally distributed**. Many features exhibit skewed patterns, particularly right-skewed distributions, where the data has a long tail toward higher values. This suggests the need for further statistical normality tests to validate the distribution shape before applying parametric methods or models that assume normality.

# #### Univariate Distribution Check for Qualitative Features

# In[5]:


categorical_features = [
    'Churn',
    'PreferredLoginDevice',
    'CityTier',
    'PreferredPaymentMode',
    'Gender',
    'PreferedOrderCat',
    'SatisfactionScore',
    'MaritalStatus',
    'Complain'
]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14, 20))
fig.suptitle('Univariate Distribution of Qualitative Features', fontsize=13, fontweight='bold', y=1.0)
axes = axes.flatten()
colors = ['#a8e6cf','#81c784','#4caf50','#388e3c','#2e7d32','#1b5e20','#0d3b1a']


for i, col in enumerate(categorical_features):
    sns.countplot(data=df, x=col, ax=axes[i], palette= colors)
    axes[i].set_title(f'Bar Plot of {col}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    axes[i].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.show()


# From the bar plots above, we can observe the distribution of categories within each categorical feature. It allows us to identify dominant classes, detect imbalances, and understand how each feature is spread across the dataset.

# ### Data preprocessing
# Data preprocessing is the process of cleaning, transforming, and organizing raw data into a usable format before applying machine learning models or conducting analysis.

# #### Handling Data Duplicates

# In[6]:


df.duplicated().sum()


# There are no duplicate rows in dataset.

# #### Missing Value

# In[7]:


(df.isna().sum() / df.shape[0] * 100).round(2).astype(str) + '%'


# In[8]:


cols_with_missing = df.columns[df.isna().any()]

missing_df = df[cols_with_missing].isna().astype(int)

missing_corr = missing_df.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(missing_corr, annot=True, cmap='Greens', fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Between Missing Values")
plt.show()


# All features with missing value have correlation values -0.05. Their missingness is independent of other columns.

# #### Casting Data Types

# #### Categorization of Dataset Attributes by Data Type
# | **Attribute**                   | **Statistical Type**      | **Explanation**                                                   |
# | ------------------------------- | ------------------------- | ----------------------------------------------------------------- |
# | **CustomerID**                  | Qualitative – Nominal     | Unique identifier, carries no numerical or ordinal meaning        |
# | **Churn**                       | Qualitative – Nominal     | Binary category (1 = churned, 0 = not churned)      |
# | **Tenure**                      | Quantitative – Continuous | Duration measured in time, can be fractional (e.g., 12.5 months)  |
# | **PreferredLoginDevice**        | Qualitative – Nominal     | Device types like Mobile/Desktop, no inherent order               |
# | **CityTier**                    | Qualitative – Ordinal     | City classification with inherent ranking (e.g., Tier 1 > Tier 3) |
# | **WarehouseToHome**             | Quantitative – Continuous | Distance, measurable and can include decimals                     |
# | **PreferredPaymentMode**        | Qualitative – Nominal     | Categories like Card, COD, etc., there are no ranking             |
# | **Gender**                      | Qualitative – Nominal     | Categories (Male, Female), not ordered                            |
# | **HourSpendOnApp**              | Quantitative – Continuous | Time in hours, can be fractional                                  |
# | **NumberOfDeviceRegistered**    | Quantitative – Discrete   | Count of devices, whole numbers only                              |
# | **PreferedOrderCat**            | Qualitative – Nominal     | Product category preference, no meaningful order                  |
# | **SatisfactionScore**           | Qualitative – Ordinal     | Rating scale, ordered values (e.g., 1 to 5)                       |
# | **MaritalStatus**               | Qualitative – Nominal     | Categories (Single, Married, etc.), no ranking                    |
# | **NumberOfAddress**             | Quantitative – Discrete   | Count of addresses, integers only                                 |
# | **Complain**                    | Qualitative – Nominal     | Binary indicator (1 = yes, 0 = no), no ordinal meaning            |
# | **OrderAmountHikeFromlastYear** | Quantitative – Continuous | Percentage change, measurable and fractional                      |
# | **CouponUsed**                  | Quantitative – Discrete   | Count of coupons used, integers only                              |
# | **OrderCount**                  | Quantitative – Discrete   | Number of orders placed, count data                               |
# | **DaySinceLastOrder**           | Quantitative – Discrete   | Number of days since last order, integers only                    |
# | **CashbackAmount**              | Quantitative – Continuous | Average cashback, decimal values possible                         |

# In[9]:


df.dtypes


# In[10]:


df['CustomerID'] = df['CustomerID'].astype(str)
df['Churn'] = df['Churn'].astype(str)
df['CityTier'] = df['CityTier'].astype(str)
df['Complain'] = df['Complain'].astype(str)
df['SatisfactionScore'] = df['SatisfactionScore'].astype(str)


# Although CustomerID is stored as a numeric type, it represents a unique identifier rather than a value for mathematical operations
# 'Churn' typically indicates whether a customer has left (Yes/No or 1/0). It's a categorical target variable, so convert it to 'category' type.
# Convert 'CityTier' to category because it represents discrete levels or segments (e.g., Tier 1, 2, 3)
# Convert 'Complain' to category, since it represents categories, not quantities.
# Convert 'SatisfactionScore' to category since it represents ordinal ratings (e.g., 1 to 5)


# #### Handling Inconsistent Variable

# Inspect the unique values to detect inconsistencies.

# In[11]:


df['PreferredLoginDevice'].value_counts()


# In[12]:


df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace({'Phone': 'Mobile Phone'})


# In[13]:


df['PreferredLoginDevice'].value_counts()


# In `PreferredLoginDevice` column, there were two labels that likely referred to the same category:
# 
# * **Mobile Phone** 
# * **Phone** (likely meant to be the same as "Mobile Phone")
# 
# To ensure consistency, replaced **Phone** with **Mobile Phone**:
# 
# This operation merges the inconsistent label into a standardized one. After this step, `Mobile Phone` now includes all users who were previously labeled as **Phone**.

# In[14]:


df['PreferredPaymentMode'].value_counts()


# In[15]:


df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace({'COD': 'Cash on Delivery'})
df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace({'CC': 'Credit Card'})


# In[16]:


df['PreferredPaymentMode'].value_counts()


# In the `PreferredPaymentMode` column, there were some labels that likely referred to the same category:
# 
# * **Cash on Delivery** 
# * **COD** (likely meant to be the same as **Cash on Delivery**)
# 
# To ensure consistency, replaced **COD** with **Cash on Delivery**:
# 
# * **Credit Card**
# * **CC** (likely meant to be the same as "Credit Card")
# 
# To ensure consistency, **CC** will be replace with **Credit Card**

# In[17]:


df['PreferedOrderCat'].value_counts()


# In[18]:


df['PreferedOrderCat'] = df['PreferedOrderCat'].replace({'Mobile': 'Mobile Phone'})


# In[19]:


df['PreferedOrderCat'].value_counts()


# In the `PreferedOrderCat` column, there were two labels that likely referred to the same category:
# 
# * **Mobile Phone**
# * **Mobile** (likely meant to be the same as **Mobile Phone**)
# 
# To ensure consistency, **Mobile** will be replace with **Mobile Phone**. This operation merges the inconsistent label into a standardized one. After this step, **Mobile Phone** now includes all users who were previously labeled as **Mobile**.

# #### Handling Outlier

# In[20]:


numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"\nColumn: {col}")
    print(f"Outlier: {len(outliers)}")
    plt.figure(figsize=(8, 1.5))
    sns.boxplot(data=df, x=col, color="#247c6d")
    plt.title(f"Boxplot - {col}")
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()


# The outliers are still close to the boxplot line or may reflect important customer behavior, so they are kept.

# #### Create New Column

# In[21]:


df['CountOfAddress'] = pd.qcut(df['NumberOfAddress'], q=4)
df


# This makes it easier to visualize and compare patterns in a more understandable way.

# In[22]:


df['CountOfAddress'].unique()


# In[23]:


interval_to_label = {
    pd.Interval(0.999, 2.0, closed='right'): '1–2',
    pd.Interval(2.0, 3.0, closed='right'): '3',
    pd.Interval(3.0, 6.0, closed='right'): '4–6',
    pd.Interval(6.0, 22.0, closed='right'): '7+'
}

df['CountOfAddress'] = df['CountOfAddress'].map(interval_to_label)
df


# In[24]:


df.drop(columns='NumberOfAddress', inplace=True)
df['CountOfAddress'] = df['CountOfAddress'].astype(str)
df
# The NumberOfAddress column is dropped because a binned version, called CountOfAddress, has already been created.


# ### Data Analysis

# #### Churn Distribution

# In[25]:


churn_counts = df['Churn'].value_counts()
labels = ['Not Churn', 'Churn']
colors = ['#1d665d', '#b2d98b']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.pie(churn_counts, labels=labels, colors=colors, explode=(0, 0.1),
        autopct='%1.1f%%', startangle=90, shadow=True)
plt.title('Customer Churn Proportion')

plt.subplot(1, 2, 2)
bars = plt.bar(labels, churn_counts, color=colors)
plt.title('Customer Churn Distribution')
plt.ylabel('Number of Customers')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        yval + max(churn_counts)*0.01,
        f'{yval:,}',                    
        ha='center',
        va='bottom',
        fontsize=10,
    )

plt.tight_layout()
plt.show()


# The charts show that **16.8% of customers have churned**, while **83.2% not churn**. Although the churn rate is relatively low, it still represents a significant number of customers. This emphasizes the need for businesses to understand the causes of churn and take early action to retain at-risk customers.

# #### Customer Demographics

# In[26]:


df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce')

demographics_cols = [
    'CityTier',
    'Gender',
    'MaritalStatus',
    'CountOfAddress',
]

colors = ['#81c784','#4caf50','#388e3c','#2e7d32','#1b5e20','#0d3b1a']


plt.figure(figsize=(22, 6 * len(demographics_cols)))

for i, col in enumerate(demographics_cols):
    row_idx = i * 3

    order = df[col].value_counts().sort_values(ascending=False).index


    plt.subplot(len(demographics_cols), 3, row_idx + 1)
    ax = sns.countplot(data=df, x=col, hue='Churn', palette=colors, order=order)
    plt.title(f'Customer Churn Distribution by {col}')
    plt.xticks(rotation=45)

import matplotlib.patches as mpatches

for p in ax.patches:
    if isinstance(p, mpatches.Rectangle):  
        count = int(p.get_height())
        if count > 0:
            ax.annotate(f'{count}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=9, color='black',
                        xytext=(0, 5),
                        textcoords='offset points')
            
    plt.subplot(len(demographics_cols), 3, row_idx + 2)
    value_counts = df[col].value_counts()
    plt.pie(value_counts, labels=value_counts.index.astype(str), autopct='%1.1f%%',
            colors=colors, startangle=90, wedgeprops={'edgecolor': 'black'})
    plt.title(f'Proportions of {col}')
    plt.axis('equal')


    plt.subplot(len(demographics_cols), 3, row_idx + 3)
    data = df.groupby(df[col].astype(str), observed=False)['Churn'].mean().reset_index(name='Churn')
    data = data.sort_values(by='Churn', ascending=False)
    overall_churn_rate = df['Churn'].mean()

    ax = sns.barplot(data=data, y=col, x='Churn', color='darkcyan', orient='h')

    for p in ax.patches:
        ax.annotate(f'{p.get_width() * 100:.2f}%',
                    (p.get_width(), p.get_y() + p.get_height() / 2),
                    ha='left', va='center',
                    xytext=(5, 0),
                    textcoords='offset points', color='black', fontsize=9, fontweight='bold')

    ax.axvline(x=overall_churn_rate, color='black', linestyle='--', label=f'Overall: {overall_churn_rate * 100:.2f}%')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
    plt.xlim(0, data['Churn'].max() + 0.05)
    plt.xlabel('Churn Rate')
    plt.title(f'Churn Rate by {col}')
    plt.legend()

plt.suptitle('Distribution, Proportions, and Churn Rate for Each Customer Demographic Feature', fontsize=18, fontweight='bold', color='black')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# In[27]:


demographicnum_cols = [
    'NumberOfDeviceRegistered',
    'WarehouseToHome'
]

colors = ['#1d665d', '#b2d98b']


for col in demographicnum_cols:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    fig.suptitle(f"Distribution of Customer Demographic Features by Churn Status", fontsize=18, fontweight='bold', color='black')


    ax_box = axes[0]
    sns.boxplot(data=df, x=col, hue='Churn', palette= colors, ax=ax_box)
    ax_box.set_title(f"{col} - Boxplot by Churn", fontsize=12)
    ax_box.set_xlabel(col)
    ax_box.set_ylabel("Distribution")


    median_not_churn = df[df['Churn'] == 0][col].median()
    median_churn = df[df['Churn'] == 1][col].median()


    info_text = (
        f"Median (No Churn): {median_not_churn:.2f}\n"
        f"Median (Churn): {median_churn:.2f}"
    )
    ax_box.text(0.98, 0.05, info_text, transform=ax_box.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))


    ax_hist = axes[1]
    sns.histplot(data=df, x=col, hue='Churn', kde=True, palette=colors, bins=30, element='step', ax=ax_hist)
    ax_hist.set_title(f"{col} - Histogram by Churn", fontsize=12)
    ax_hist.set_xlabel(col)
    ax_hist.set_ylabel("Count")


    plt.tight_layout()
    plt.show()


# * City Tier : Most customers are from City Tier 1, followed by Tier 3, and then Tier 2. The churn rate in Tier 3 is over 50% higher than in Tier 1. Showing that customers in smaller or less developed cities are more likely to churn, possibly due to service accessibility, pricing sensitivity, or local alternatives.
#   
# * Gender : There are more male customers than female. The churn rate is slightly higher for male customers, but the difference is not large. This suggests that gender has limited influence on churn behavior.
#   
# * Marital Status : Most customers are married, followed by single and divorced. Single customers churn more often, while married customers are the most loyal.
#   
# * Count of Address: Most customers have 1–2 or 4–6 addresses, with fewer customers having 3 or 7+. The churn rate increases as the number of addresses increases, especially for those with 7+ addresses (22.04%), which is significantly above the overall churn rate of 16.84%. This may indicate that customers with frequent address changes are less stable and more likely to churn.
# 
# * Number of Devices Registered: The distribution is centered around 3 to 4 devices for both churned and retained customers. However, churn customers are less frequent across all device counts, and no strong pattern is observed, suggesting this feature may be less impactful for churn prediction.
# 
# * Warehouse To Home: This feature represents the distance between the warehouse and the customer's home. The median distance is slightly higher for churned customers (15) than for non-churned customers (13), as shown in the boxplot. The histogram shows that churned customers are somewhat more spread out across higher distance values. This suggests that longer distances may slightly increase the risk of churn, possibly due to delivery challenges or perceived inconvenience.
# 
# >City Tier, Marital Status, and Count of Address show strong relationships with churn and should be prioritized in modeling. Gender, Number of Devices Registered, and Warehouse To Home have weaker impact but may still add value when combined with other features.

# #### Customer Behaviour/ App usage

# In[28]:


df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce')

behaviour_cols = [
    'PreferredLoginDevice',
    'PreferredPaymentMode',
    'PreferedOrderCat',
]
colors = ['#81c784','#4caf50','#388e3c','#2e7d32','#1b5e20','#0d3b1a']

plt.figure(figsize=(22, 6 * len(demographics_cols)))  

for i, col in enumerate(behaviour_cols):
    row_idx = i * 3

    order = df[col].value_counts().sort_values(ascending=False).index


    plt.subplot(len(behaviour_cols), 3, row_idx + 1)
    ax = sns.countplot(data=df, x=col, hue='Churn', palette=colors, order=order)
    plt.title(f'Customer Churn Distribution by {col}')
    plt.xticks(rotation=45)

    for p in ax.patches:
        count = int(p.get_height())
        if count > 0:
            ax.annotate(f'{count}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 5), 
                        textcoords='offset points', color='black', fontsize=9, fontweight='bold')


    plt.subplot(len(behaviour_cols), 3, row_idx + 2)
    value_counts = df[col].value_counts()
    plt.pie(value_counts, labels=value_counts.index.astype(str), autopct='%1.1f%%',
            colors=colors, startangle=90, wedgeprops={'edgecolor': 'black'})
    plt.title(f'Proportions of {col}')
    plt.axis('equal')


    plt.subplot(len(behaviour_cols), 3, row_idx + 3)
    data = df.groupby(df[col].astype(str), observed=False)['Churn'].mean().reset_index(name='Churn')
    data = data.sort_values(by='Churn', ascending=False)
    overall_churn_rate = df['Churn'].mean()

    ax = sns.barplot(data=data, y=col, x='Churn', color='darkcyan', orient='h')

    for p in ax.patches:
        ax.annotate(f'{p.get_width() * 100:.2f}%',
                    (p.get_width(), p.get_y() + p.get_height() / 2),
                    ha='left', va='center',
                    xytext=(5, 0),
                    textcoords='offset points', color='black', fontsize=9, fontweight='bold')

    ax.axvline(x=overall_churn_rate, color='black', linestyle='--', label=f'Overall: {overall_churn_rate * 100:.2f}%')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
    plt.xlim(0, data['Churn'].max() + 0.05)
    plt.xlabel('Churn Rate')
    plt.title(f'Churn Rate by {col}')
    plt.legend()

plt.suptitle('Distribution, Proportions, and Churn Rate for Each Customer Behaviour Feature', fontsize=18, fontweight='bold', color='black')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# In[29]:


behaviournum_cols = [
    'HourSpendOnApp','Tenure'
]
colors = ['#1d665d', '#b2d98b']

for col in behaviournum_cols:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    fig.suptitle(f"Distribution of Customer Behaviour Features by Churn Status", fontsize=18, fontweight='bold', color='black')


    ax_box = axes[0]
    sns.boxplot(data=df, x=col, hue='Churn', palette= colors, ax=ax_box)
    ax_box.set_title(f"{col} - Boxplot by Churn", fontsize=12)
    ax_box.set_xlabel(col)
    ax_box.set_ylabel("Distribution")


    median_not_churn = df[df['Churn'] == 0][col].median()
    median_churn = df[df['Churn'] == 1][col].median()


    info_text = (
        f"Median (No Churn): {median_not_churn:.2f}\n"
        f"Median (Churn): {median_churn:.2f}"
    )
    ax_box.text(0.98, 0.05, info_text, transform=ax_box.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))


    ax_hist = axes[1]
    sns.histplot(data=df, x=col, hue='Churn', kde=True, palette=colors, bins=30, element='step', ax=ax_hist)
    ax_hist.set_title(f"{col} - Histogram by Churn", fontsize=12)
    ax_hist.set_xlabel(col)
    ax_hist.set_ylabel("Count")


    plt.tight_layout()
    plt.show()


# * Preferred Login Device : Most customers use a mobile phone to log in followed by computers. This difference is above the overall churn rate (13.2%), suggesting that customers logging in via computer are more likely to churn.
# 
# * Preferred Payment Mode : Most customers prefer debit cards and credit cards. However, cash on delivery (18.86%) and e-wallet (18.02%) users show higher churn rates, well above the average churn rate of 13.2%. Customers using convenient, flexible payment methods might be less loyal and easily switch platforms.
# 
# * Preferred Order Category :  Customers who mostly buy Mobile Phones churn the most (22.72%), far above the average. Those who buy Grocery (4.89%) or Laptop & Accessories (7.28%) churn far less. The large gaps between categories show this is a very strong signal of churn risk.
# 
# * Time Spent on App : Users spending 3–5 hours on the app have a higher churn rate.
# 
# * Tenure : Customers who churn tend to have significantly shorter tenure compared to those who stay. The boxplot shows that the median tenure for churned customers is only 1 month, while for non-churned customers it is 10 months. The histogram further confirms that customers with longer tenure are less likely to churn. This indicates that customer longevity is a strong signal of churn risk.
# 
# >Customers who use computers to log in, pay with cash or e-wallets, often buy mobile phones, or spend 3–5 hours on the app are more likely to churn. Those with shorter tenure (median of 1 month) also show significantly higher churn, emphasizing the importance of customer longevity. These differences are quite large, so they are important to include in a churn prediction model to help identify and retain high-risk customers.

# #### Customer Purchase History

# In[30]:


purchase_cols = [
    'OrderAmountHikeFromlastYear',
    'CouponUsed',
    'OrderCount',
    'DaySinceLastOrder',
    'CashbackAmount'
]
colors = ['#1d665d', '#b2d98b']

for col in purchase_cols:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    fig.suptitle(f"Distribution of Customer Purchase History Features by Churn Status", fontsize=18, fontweight='bold', color='black')


    ax_box = axes[0]
    sns.boxplot(data=df, x=col, hue='Churn', palette= colors, ax=ax_box)
    ax_box.set_title(f"{col} - Boxplot by Churn", fontsize=12)
    ax_box.set_xlabel(col)
    ax_box.set_ylabel("Distribution")


    median_not_churn = df[df['Churn'] == 0][col].median()
    median_churn = df[df['Churn'] == 1][col].median()


    info_text = (
        f"Median (No Churn): {median_not_churn:.2f}\n"
        f"Median (Churn): {median_churn:.2f}"
    )
    ax_box.text(0.98, 0.05, info_text, transform=ax_box.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))


    ax_hist = axes[1]
    sns.histplot(data=df, x=col, hue='Churn', kde=True, palette=colors, bins=30, element='step', ax=ax_hist)
    ax_hist.set_title(f"{col} - Histogram by Churn", fontsize=12)
    ax_hist.set_xlabel(col)
    ax_hist.set_ylabel("Count")


    plt.tight_layout()
    plt.show()


# * Order Amount Hike From Last Year: The median is similar for both churned and retained customers (15 vs. 14), with similar spread. This feature may not be very useful for predicting churn.
#   
# * Coupon Used: Churned customers tend to use slightly fewer coupons. Although the median is the same (1), the lower spread suggests that more engaged users (using more coupons) are less likely to churn.
# 
# * Order Count: Both groups have the same median number of orders (2), but churned customers have fewer high-order outliers. This may suggest that frequent buyers are more likely to stay.
# 
# * Days Since Last Order: Churned users usually made their last purchase more recently (median = 2) than retained ones (median = 4). This may indicate they churn shortly after their final order.
# 
# * Cashback Amount: Retained customers receive slightly more cashback (166 vs. 149), which may help reduce churn risk.
# 
# >Variables related to customer purchasing behavior—such as Coupon Used, Order Count, Cashback Amount, and Days Since Last Order—demonstrate strong differentiation between churned and retained customers, making them highly valuable for predictive modeling.

# #### Customer Feedback/ Satisfaction

# In[31]:


df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce')

feedback_cols = [
    'SatisfactionScore',
    'Complain'
]
colors = ['#81c784','#4caf50','#388e3c','#2e7d32','#1b5e20','#0d3b1a']

plt.figure(figsize=(22, 6 * len(demographics_cols)))  

for i, col in enumerate(feedback_cols):
    row_idx = i * 3

    order = df[col].value_counts().sort_values(ascending=False).index


    plt.subplot(len(feedback_cols), 3, row_idx + 1)
    ax = sns.countplot(data=df, x=col, hue='Churn', palette=colors, order=order)
    plt.title(f'Customer Churn Distribution by {col}')
    plt.xticks(rotation=45)

    for p in ax.patches:
        count = int(p.get_height())
        if count > 0:
            ax.annotate(f'{count}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 5), 
                        textcoords='offset points', color='black', fontsize=9, fontweight='bold')


    plt.subplot(len(feedback_cols), 3, row_idx + 2)
    value_counts = df[col].value_counts()
    plt.pie(value_counts, labels=value_counts.index.astype(str), autopct='%1.1f%%',
            colors=colors, startangle=90, wedgeprops={'edgecolor': 'black'})
    plt.title(f'Proportions of {col}')
    plt.axis('equal')


    plt.subplot(len(feedback_cols), 3, row_idx + 3)
    data = df.groupby(df[col].astype(str), observed=False)['Churn'].mean().reset_index(name='Churn')
    data = data.sort_values(by='Churn', ascending=False)
    overall_churn_rate = df['Churn'].mean()

    ax = sns.barplot(data=data, y=col, x='Churn', color='darkcyan', orient='h')

    for p in ax.patches:
        ax.annotate(f'{p.get_width() * 100:.2f}%',
                    (p.get_width(), p.get_y() + p.get_height() / 2),
                    ha='left', va='center',
                    xytext=(5, 0),
                    textcoords='offset points', color='black', fontsize=9, fontweight='bold')

    ax.axvline(x=overall_churn_rate, color='black', linestyle='--', label=f'Overall: {overall_churn_rate * 100:.2f}%')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
    plt.xlim(0, data['Churn'].max() + 0.05)
    plt.xlabel('Churn Rate')
    plt.title(f'Churn Rate by {col}')
    plt.legend()

plt.suptitle('Distribution, Proportions, and Churn Rate for Each Customer Feedback Feature', fontsize=18, fontweight='bold', color='Black')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# * Satisfaction Score : Customers with higher satisfaction scores tend to churn more, especially those with a score of 5, where the churn rate reaches 18.51%, well above the overall churn rate of 13.20%. Meanwhile, those with low satisfaction scores (like 1 or 2) have churn rates below 10%.
#   
# * Complain : Customers who filed complaints (Complain = 1) show a churn rate of 26.17%, double the overall churn rate. Meanwhile, those who did not complain have a churn rate of only 8.27%. This indicates that complaint are a clear warning sign of potential churn.
# 
# >Customers who complain or have very high satisfaction scores are more likely to churn. These factors are significant and should be considered in churn prediction models to help businesses respond early and reduce customer loss.

# ### Correlation Matrix

# Based on the previous exploratory analysis, the data does not follow a normal distribution. Therefore, a non-parametric test was selected as it is more appropriate for this type of distribution.

# In[32]:


X = df.drop(columns='Churn').select_dtypes(include='number')

corr = X.corr(method='spearman')

plt.figure(figsize=(10, 7))
sns.heatmap(corr, vmin=-1, vmax=1, annot=True, cmap='Greens', mask=np.triu(np.ones_like(corr, dtype=bool)))
plt.title('Spearman Correlation Matrix')
plt.show()


# >The Spearman correlation matrix shows no strong multicollinearity among numerical features, which is favorable for modeling. However, there is a moderate correlations to note like Order Count and Coupon Used with correlation value 0.7. These suggest some relationship but not enough to warrant immediate feature removal. Overall, all numerical features can be safely used in modeling without major concerns for multicollinearity.

# In[33]:


features = [
    'Tenure', 'WarehouseToHome', 'HourSpendOnApp',
    'NumberOfDeviceRegistered',
    'OrderAmountHikeFromlastYear', 'CouponUsed',
    'OrderCount', 'DaySinceLastOrder', 'CashbackAmount'
]

results = []

for feature in features:
    churned = df[df['Churn'] == 1][feature].dropna()
    not_churned = df[df['Churn'] == 0][feature].dropna()

    stat, p = mannwhitneyu(churned, not_churned, alternative='two-sided')

    results.append({
        'Feature': feature,
        'U-Statistic': stat,
        'p-value': p,
        'Significance': 'Significant' if p < 0.05 else 'Not Significant'
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(by='p-value')

results_df


# >The goal of this test is to identify which features show statistically significant differences between churned and non-churned customers. The results of the Mann-Whitney U test indicate that several features show statistically significant differences between churned and non-churned customers, making them valuable for predictive modeling. Tenure, Day Since Last Order, Cashback Amount, Number Of Device Registered, Warehouse To Home, and Order Count have p-values below 0.05, suggesting a meaningful relationship with customer churn. These features are strong candidates to be included in churn prediction models. On the other hand, Order Amount Hike From Last Year, Hour Spend On App, and Coupon Used did not show significant differences between the two groups, implying they may have limited predictive power for churn in this context and could be deprioritized or excluded during feature selection. Overall, the statistically significant features identified can help enhance the accuracy and interpretability of churn models.

# In[34]:


cat_features = [
    'PreferredLoginDevice', 'CityTier',
    'PreferredPaymentMode', 'Gender',
    'PreferedOrderCat', 'SatisfactionScore',
    'Complain', 'CountOfAddress'
]

results = []

for col in cat_features:
    table = pd.crosstab(df[col], df['Churn'])
    chi2, p, _, _ = chi2_contingency(table)

    significance = "Significant" if p < 0.05 else "Not Significant"

    results.append({
        'Feature': col,
        'Chi²': chi2,
        'p-value': p,
        'Significance': significance
    })

chi2_df = pd.DataFrame(results)
chi2_df = chi2_df.sort_values(by='p-value')

chi2_df


# >The Chi-square test shows that all categorical features are significantly associated with churn. Key drivers include Complain, Preferred Order Cat, and Satisfaction Score, which have the strongest impact. Other relevant features are Preferred Payment Mode, City Tier, Count Of Address, Preferred LoginDevice, and Gender. All should be considered for churn modeling.

# In[35]:


df.to_csv('e_comm_cleaned.csv', index=False)


# ## Build Model

# #### Flowchart: End-to-End Modeling Process

# ![terbaru bgt finpro flowchart 1.drawio.png](attachment:9416c9c9-6946-4ef4-98ce-4a7f1bcb3017.png)
# ![terbaru flo finpro 2.drawio.png](attachment:4266fd56-b597-4c09-8950-a1f9555ad85a.png)

# #### Modeling Flowchart
# 
# This end-to-end modeling flow starts by **loading the dataset** and **splitting it into training and testing sets (80/20)**. Missing values are handled using either **Median Imputer** or **Iterative Imputer**. **Feature selection** is performed using methods like **SelectFromModel**, **SelectKBest**, or **SelectPercentile** to reduce dimensionality. To address **class imbalance**, **resampling techniques** such as **Random Over Sampling (ROS)**, **Random Under Sampling (RUS)**, and **SMOTE** are applied within the modeling pipeline. All **preprocessing model combinations** are evaluated, and the **best pipeline** is selected.   **Hyperparameter tuning** is then performed to **optimize model performance**. The optimized model is **fitted on the training set** and **evaluated on the test set**. The final step involves analyzing **performance metrics** and **feature importance** to assess the model’s **effectiveness** and **interpretability**.

# ### Modeling

# In[36]:


X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=79)


# ### Feature Selection (SelectFromModel) + Median Imputer + With & Without Sampling

# In[38]:


binary_encode_cols = ['PreferredPaymentMode', 'PreferedOrderCat', 'MaritalStatus','CountOfAddress']
onehot_encode_cols = ['PreferredLoginDevice', 'Gender']
numeric_cols = [
    'Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
    'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
    'DaySinceLastOrder', 'CashbackAmount'
]


preprocessor = ColumnTransformer(
    transformers=[

        ('Numerical 1',
         ImbPipeline([
             ('Imputer 1', SimpleImputer(strategy='median')),
             ('Scaler', RobustScaler())
         ]),
         numeric_cols),

        ('Categorical 1',
         ImbPipeline([
             ('Encoder 1', ce.BinaryEncoder())
         ]),
         binary_encode_cols),

         ('Categorical 2',
         ImbPipeline([
             ('Encoder 2', OneHotEncoder(drop="first"))
         ]),
         onehot_encode_cols),

        ('Categorical 3',
         ImbPipeline([
             ('Encoder 3', OrdinalEncoder(categories=[['1','2','3','4','5']]))
         ]),
         ['SatisfactionScore']),

         ('Categorical 4',
         ImbPipeline([
             ('Encoder 4', OrdinalEncoder(categories=[['3','2','1']]))
         ]),
         ['CityTier']),
    ],
    remainder='drop' 
)


tree = DecisionTreeClassifier(random_state=79)
logreg = LogisticRegression(random_state=79)
xgb = XGBClassifier(random_state=79)
rf = RandomForestClassifier(random_state=79)
knn = KNeighborsClassifier() 
lgbm = LGBMClassifier(random_state=79)
svm = SVC(probability=True, random_state=79)

sampling_strategies = {
    "No Resampling": None,
    "SMOTE": SMOTE(random_state=79),
    "RandomOverSampler": RandomOverSampler(random_state=79),
    "RandomUnderSampler": RandomUnderSampler(random_state=79),
}


for sampler_name, sampler in sampling_strategies.items():
    print(f"Running pipeline with: {sampler_name}")

    steps = [('preprocessor', preprocessor)]

    if sampler is not None:
        steps.append(('sampler', sampler))

    steps += [

        ('feature_selection', SelectFromModel(estimator=XGBClassifier())),
        ('classifier', rf)]  

    imbpipeline = ImbPipeline(steps=steps)


# In[39]:


imbpipeline.fit(X_train,y_train)


# In[40]:


tree = DecisionTreeClassifier(random_state=79)
logreg = LogisticRegression(random_state=79)
xgb = XGBClassifier(random_state=79)
rf = RandomForestClassifier(random_state=79)
knn = KNeighborsClassifier()
lgbm = LGBMClassifier(random_state=79)
svm = SVC(probability=True, random_state=79)

list_model = [tree, logreg, xgb, rf, knn, lgbm, svm]
nama_model = ['Decision Tree', 'Logistic Regression', 'XGBoost', 'Random Forest', 'K-Nearest Neighbors', 'LGBMClassifier', 'SVC']

sampling_strategies = {
    "No Resampling": None,
    "SMOTE": SMOTE(random_state=79),
    "RandomOverSampler": RandomOverSampler(random_state=79),
    "RandomUnderSampler": RandomUnderSampler(random_state=79),
}

results_fromModel = []

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=79)
f2 = make_scorer(fbeta_score, beta=2)

for sampler_name, sampler in sampling_strategies.items():
    for i in range(len(list_model)):
        model = list_model[i]
        model_name = nama_model[i]

        steps = [('preprocessor', preprocessor)]

        if sampler is not None:
            steps.append(('sampler', sampler))

        steps += [
            ('feature_selection', SelectFromModel(estimator=xgb)),
            ('classifier', model)
        ]

        pipeline = ImbPipeline(steps=steps)

        result = cross_validate(pipeline, X_train, y_train,
                                scoring=f2,
                                cv=skfold,
                                return_train_score=False)

        results_fromModel.append({
            'Sampler': sampler_name,
            'Model': model_name,
            'Avg F2 Score': result['test_score'].mean(),
            'F2 Score Std': result['test_score'].std(),
            'Avg Fit Time': result['fit_time'].mean(),
            'Avg Score Time': result['score_time'].mean()
        })

df_result = pd.DataFrame(results_fromModel)
df_result = df_result.sort_values(by='Avg F2 Score', ascending=False).reset_index(drop=True)
display(df_result)


# #### Top 3 Model Performance with Feature Selection SelectFromModel + Median Imputer
# 
# | Sampler              | Model          | Avg F2 Score | F2 Score Std | Avg Fit Time | Avg Score Time |
# |----------------------|----------------|--------------|---------------|---------------|----------------|
# | RandomOverSampler    | LGBMClassifier | **0.6893**   | 0.0303        | 0.1287        | 0.011         |
# | RandomOverSampler    | XGBoost        | **0.6885**   | 0.0257        | 0.1487        | 0.010         |
# | RandomUnderSampler   | Random Forest  | **0.6866**   | 0.0157        | 0.2681        | 0.036         |
# 
# #### Insights
# 
# - **LGBMClassifier with RandomOverSampler** achieved the **highest average F2 score (0.6893)**, suggesting it slightly outperforms others in capturing the positive class.
# - **XGBoost** follows closely, also using **RandomOverSampler**, and provides a good balance between F2 score.
# - **Random Forest with RandomUnderSampler** showed the **most stable results** (lowest F2 score std: 0.0157), indicating consistent behavior across cross-validation folds.
# - Overall, **RandomOverSampler combinations dominated the top results**, reaffirming that oversampling may be more effective than undersampling for this dataset.

# ### Feature Selection (SelectKBest) + Median Imputer + With & Without Sampling

# In[41]:


for k in [5, 10, 15, 18, 21, 24]:
    for sampler_name, sampler in sampling_strategies.items():
        print(f"Running: {sampler_name} | k={k}")

        steps = [('preprocessor', preprocessor)]

        if sampler is not None:
            steps.append(('sampler', sampler))

        steps += [
            ('feature_selection', SelectKBest(score_func=f_classif, k=k)),
            ('classifier', rf)
        ]

        imbpipeline = ImbPipeline(steps=steps)


# In[42]:


imbpipeline.fit(X_train,y_train)


# In[43]:


len(imbpipeline.named_steps['preprocessor'].get_feature_names_out())


# The pipeline has generated 24 final features after all preprocessing steps (including encoding, imputation, scaling). These 24 are the features actually passed into the model.

# In[44]:


results_kbest = []

for k in [5, 10, 15, 18, 21, 24]:
    for sampler_name, sampler in sampling_strategies.items():
        for i in range(len(list_model)):
            model = list_model[i]
            model_name = nama_model[i]

            steps = [('preprocessor', preprocessor)]

            if sampler is not None:
                steps.append(('sampler', sampler))

            steps += [
                ('feature_selection', SelectKBest(score_func=f_classif, k=k)),
                ('classifier', model)
            ]

            pipeline = ImbPipeline(steps=steps)

            result = cross_validate(pipeline, X_train, y_train,
                                    scoring=f2,
                                    cv=skfold,
                                    return_train_score=False)

            results_kbest.append({
                'K Features': k,
                'Sampler': sampler_name,
                'Model': model_name,
                'Avg F2 Score': result['test_score'].mean(),
                'F2 Score Std': result['test_score'].std(),
                'Avg Fit Time': result['fit_time'].mean(),
                'Avg Score Time': result['score_time'].mean()
            })

df_kbest = pd.DataFrame(results_kbest)
df_kbest = df_kbest.sort_values(by='Avg F2 Score', ascending=False).reset_index(drop=True)
display(df_kbest.head(30))


# #### Top 3 Model Performance with Feature Selection SelectKBest + Median Imputer  
# 
# | K Features | Sampler            | Model         | Avg F2 Score | F2 Score Std | Avg Fit Time | Avg Score Time |
# |------------|--------------------|---------------|--------------|---------------|----------------|------------------|
# | 24         | RandomOverSampler  | LGBMClassifier| **0.8514**   | 0.0103        | 0.128         | 0.0111           |
# | 24         | RandomOverSampler  | XGBoost       | **0.8500**   | 0.0155        | 0.185        | 0.0162           |
# | 21         | RandomOverSampler  | XGBoost       | **0.8425**   | 0.0216        | 0.215         | 0.0177           |
# 
# 
# #### Insights
# 
# - **LGBMClassifier with 24 features and RandomOverSampler** achieved the **highest average F2 score (0.8514)**, making it the best performer in this configuration.
# - **XGBoost** also showed strong performance at `k=24`, nearly matching LGBM’s F2 score.
# - Reducing the number of features to 21 for XGBoost led to a **small drop in performance (0.8425)**, but with a **higher training time**, suggesting that fewer features didn’t necessarily improve efficiency.
# - LGBMClassifier not only topped the chart in F2 score, but also had the **lowest standard deviation (0.0103)**, indicating highly stable results across validation folds.
# - All top models used **RandomOverSampler**, reinforcing its consistent advantage in handling class imbalance effectively.

# ### Feature Selection (SelectPercentile) + Median Imputer + With & Without Sampling

# In[66]:


for p in [30, 50, 70, 90, 95]:
    for sampler_name, sampler in sampling_strategies.items():
        print(f"Running: {sampler_name} | percentile={p}%")

        steps = [('preprocessor', preprocessor)]

        if sampler is not None:
            steps.append(('sampler', sampler))

        steps += [
            ('feature_selection', SelectPercentile(score_func=f_classif, percentile=p)),
            ('classifier', rf)
        ]

        imbpipeline = ImbPipeline(steps=steps)


# In[67]:


imbpipeline.fit(X_train,y_train)


# In[68]:


results_percentile = []


for p in [30, 50, 70, 90, 95]:
    for sampler_name, sampler in sampling_strategies.items():
        for i in range(len(list_model)):
            model = list_model[i]
            model_name = nama_model[i]

            steps = [('preprocessor', preprocessor)]

            if sampler is not None:
                steps.append(('sampler', sampler))

            steps += [
                ('feature_selection', SelectPercentile(score_func=f_classif, percentile=p)),
                ('classifier', model)
            ]

            pipeline = ImbPipeline(steps=steps)


            result = cross_validate(pipeline, X_train, y_train,
                                    scoring=f2,
                                    cv=skfold,
                                    return_train_score=False)

            results_percentile.append({
                'Percentile': p,
                'Sampler': sampler_name,
                'Model': model_name,
                'Avg F2 Score': result['test_score'].mean(),
                'F2 Score Std': result['test_score'].std(),
                'Avg Fit Time': result['fit_time'].mean(),
                'Avg Score Time': result['score_time'].mean()
            })


df_percentile = pd.DataFrame(results_percentile)
df_percentile = df_percentile.sort_values(by='Avg F2 Score', ascending=False).reset_index(drop=True)
display(df_percentile.head(30))


# #### Top 3 Model Performance with Feature Selection SelectPercentile + Median Imputer  
# 
# | Percentile | Sampler            | Model          | Avg F2 Score | F2 Score Std | Avg Fit Time | Avg Score Time |
# |------------|--------------------|----------------|--------------|---------------|----------------|------------------|
# | 90        | RandomOverSampler  | XGBoost         | **0.8425**   | 0.0216        | 0.165         | 0.0139           |
# | 95        | RandomOverSampler  | XGBoost         | **0.8397**   | 0.0246        | 0.190         | 0.0161           |
# | 90         | RandomOverSampler  | LGBMClassifier | **0.8347**   | 0.0113        | 0.127        | 0.0114           |
# 
# 
# #### Insights
# 
# - **XGBoost with 90% percentile features** achieved the **highest F2 score (0.8425)** with strong consistency (std: 0.0216), making it the top performer in this configuration.
# - **LGBMClassifier with 90% features** showed a **drop in performance (0.8347)** but achieved **lowest std result (0.0113)** across folds, although it required significantly more training time.
# - Once again, **RandomOverSampler** was used in all top models, reinforcing its effectiveness in handling class imbalance.

# ### Feature Selection (SelectFromModel) + Iterative Imputer + With & Without Sampling

# In[49]:


binary_encode_cols = ['PreferredPaymentMode', 'PreferedOrderCat', 'MaritalStatus','CountOfAddress']
onehot_encode_cols = ['PreferredLoginDevice', 'Gender']
numeric_cols = [
    'Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
    'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
    'DaySinceLastOrder', 'CashbackAmount'
]


preprocessor_iterative = ColumnTransformer(
    transformers=[

        ('Numerical 1',
         ImbPipeline([
             ('Imputer 1', IterativeImputer(random_state=79)),
             ('Scaler', RobustScaler())
         ]),
         numeric_cols),

        ('Categorical 1',
         ImbPipeline([
             ('Encoder 1', ce.BinaryEncoder())
         ]),
         binary_encode_cols),

         ('Categorical 2',
         ImbPipeline([
             ('Encoder 2', OneHotEncoder(drop="first"))
         ]),
         onehot_encode_cols),

        ('Categorical 3',
         ImbPipeline([
             ('Encoder 3', OrdinalEncoder(categories=[['1','2','3','4','5']]))
         ]),
         ['SatisfactionScore']),

         ('Categorical 4',
         ImbPipeline([
             ('Encoder 4', OrdinalEncoder(categories=[['3','2','1']]))
         ]),
         ['CityTier']),
    ],
    remainder='drop' 
)


tree = DecisionTreeClassifier(random_state=79)
logreg = LogisticRegression(random_state=79)
xgb = XGBClassifier(random_state=79)
rf = RandomForestClassifier(random_state=79)
knn = KNeighborsClassifier() 
lgbm = LGBMClassifier(random_state=79)
svm = SVC(probability=True, random_state=79)

sampling_strategies = {
    "No Resampling": None,
    "SMOTE": SMOTE(random_state=79),
    "RandomOverSampler": RandomOverSampler(random_state=79),
    "RandomUnderSampler": RandomUnderSampler(random_state=79),
}


for sampler_name, sampler in sampling_strategies.items():
    print(f"Running pipeline with: {sampler_name}")

    steps = [('preprocessor', preprocessor_iterative)]

    if sampler is not None:
        steps.append(('sampler', sampler))

    steps += [
        ('feature_selection', SelectFromModel(estimator=XGBClassifier())),
        ('classifier', rf)
    ]

    imbpipeline = ImbPipeline(steps=steps)


# In[50]:


imbpipeline.fit(X_train,y_train)


# In[51]:


tree = DecisionTreeClassifier(random_state=79)
logreg = LogisticRegression(random_state=79)
xgb = XGBClassifier(random_state=79)
rf = RandomForestClassifier(random_state=79)
knn = KNeighborsClassifier()
lgbm = LGBMClassifier(random_state=79)
svm = SVC(probability=True, random_state=79)

list_model = [tree, logreg, xgb, rf, knn, lgbm, svm]
nama_model = ['Decision Tree', 'Logistic Regression', 'XGBoost', 'Random Forest', 'K-Nearest Neighbors', 'LGBMClassifier', 'SVC']


sampling_strategies = {
    "No Resampling": None,
    "SMOTE": SMOTE(random_state=79),
    "RandomOverSampler": RandomOverSampler(random_state=79),
    "RandomUnderSampler": RandomUnderSampler(random_state=79),
}


results_fromModel_iterative = []


skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=79)
f2 = make_scorer(fbeta_score, beta=2)


for sampler_name, sampler in sampling_strategies.items():
    for i in range(len(list_model)):
        model = list_model[i]
        model_name = nama_model[i]

        steps = [('preprocessor', preprocessor_iterative)]

        if sampler is not None:
            steps.append(('sampler', sampler))

        steps += [
            ('feature_selection', SelectFromModel(estimator=xgb)),
            ('classifier', model)
        ]

        pipeline = ImbPipeline(steps=steps)


        result = cross_validate(pipeline, X_train, y_train,
                                scoring=f2,
                                cv=skfold,
                                return_train_score=False)

        results_fromModel_iterative.append({
            'Sampler': sampler_name,
            'Model': model_name,
            'Avg F2 Score': result['test_score'].mean(),
            'F2 Score Std': result['test_score'].std(),
            'Avg Fit Time': result['fit_time'].mean(),
            'Avg Score Time': result['score_time'].mean()
        })


df_result_IterativeSelectFromModel = pd.DataFrame(results_fromModel_iterative)
df_result_IterativeSelectFromModel = df_result_IterativeSelectFromModel.sort_values(by='Avg F2 Score', ascending=False).reset_index(drop=True)
display(df_result_IterativeSelectFromModel)


# #### Top 3 Model Performance with SelectFromModel + Iterative Imputer  
# 
# | Sampler            | Model          | Avg F2 Score | F2 Score Std | Avg Fit Time | Avg Score Time |
# |--------------------|----------------|--------------|---------------|----------------|------------------|
# | RandomOverSampler  | LGBMClassifier | **0.7255**   | 0.0155        | 0.6456         | 0.0187           |
# | RandomUnderSampler | XGBoost        | **0.7181**   | 0.0214        | 0.3763         | 0.0125           |
# | RandomUnderSampler | LGBMClassifier | **0.7165**   | 0.0187        | 0.4401         | 0.0170           |
# 
# 
# #### Insights
# 
# - **LGBMClassifier with RandomOverSampler** achieved the **highest F2 score (0.7255)** in this setup, but also had the **longest training time**.
# - **XGBoost with RandomUnderSampler** was the **most efficient**, offering faster training (0.3763s) and scoring (0.0125s), with only a slight drop in F2 performance.
# - **LGBMClassifier with RandomUnderSampler** had the lowest F2 among the three (0.7165), but still maintained strong stability across folds.
# - Overall, models with **RandomUnderSampler** performed slightly lower in F2 but were more computationally efficient.
# - This suggests a trade-off in this configuration: **RandomOverSampler provides better F2,** while **RandomUnderSampler offers speed and simplicity.**

# ### Feature Selection (SelectKBest) + Iterative Imputer + With & Without Sampling

# In[52]:


for k in [5, 10, 15, 18, 21, 24]:
    for sampler_name, sampler in sampling_strategies.items():
        print(f"Running pipeline with: {sampler_name} | k={k}")

        steps = [('preprocessor', preprocessor_iterative)]

        if sampler is not None:
            steps.append(('sampler', sampler))

        steps += [
            ('feature_selection', SelectKBest(score_func=f_classif, k=k)),
            ('classifier', rf)
        ]

        imbpipeline = ImbPipeline(steps=steps)


# In[53]:


imbpipeline.fit(X_train,y_train)


# In[54]:


results_kbest_iterative = []

for k in [5, 10, 15, 18, 21, 24]:
    for sampler_name, sampler in sampling_strategies.items():
        for i in range(len(list_model)):
            model = list_model[i]
            model_name = nama_model[i]

            steps = [('preprocessor', preprocessor_iterative)]

            if sampler is not None:
                steps.append(('sampler', sampler))

            steps += [
                ('feature_selection', SelectKBest(score_func=f_classif, k=k)),
                ('classifier', model)
            ]

            pipeline = ImbPipeline(steps=steps)

            result = cross_validate(pipeline, X_train, y_train,
                                    scoring=f2,
                                    cv=skfold,
                                    return_train_score=False)

            results_kbest_iterative.append({
                'K Features': k,
                'Sampler': sampler_name,
                'Model': model_name,
                'Avg F2 Score': result['test_score'].mean(),
                'F2 Score Std': result['test_score'].std(),
                'Avg Fit Time': result['fit_time'].mean(),
                'Avg Score Time': result['score_time'].mean()
            })

df_kbest_iterative = pd.DataFrame(results_kbest_iterative)
df_kbest_iterative = df_kbest_iterative.sort_values(by='Avg F2 Score', ascending=False).reset_index(drop=True)
display(df_kbest_iterative.head(30))


# #### Top 3 Model Performance with SelectKBest + Iterative Imputer
# 
# | K Features | Sampler           | Model          | Avg F2 Score | F2 Score Std | Avg Fit Time | Avg Score Time |
# |------------|-------------------|----------------|--------------|---------------|----------------|------------------|
# | 24         | RandomOverSampler | XGBoost        | **0.8545**   | 0.0074        | 0.402         | 0.0157           |
# | 24         | RandomOverSampler | LGBMClassifier | **0.8506**   | 0.0194        | 0.530         | 0.0165           |
# | 21         | RandomOverSampler | XGBoost        | **0.8412**   | 0.0202        | 0.520         | 0.0172           |
# 
# #### Insights
# 
# - **XGBoost with 24 features** (RandomOverSampler) yields the **best F2 score overall (0.8545)** and shows excellent stability (lowest std).
# - **LGBMClassifier** with the same 24 features is just slightly behind, with a small trade-off in F2 (0.8506).
# - Reducing features from 24 to **21 causes a drop in F2** for XGBoost (to 0.8412).
# - RandomOverSampler again proves to be highly effective in boosting F2 performance across models.
# - Overall, **keeping 24 features + ROS + XGBoost** is the **best combo so far**, balancing performance and time cost effectively.

# ### Feature Selection (SelectPercentile) + Iterative Imputer + With & Without Sampling

# In[69]:


for p in [30, 50, 70, 90, 95]:
    for sampler_name, sampler in sampling_strategies.items():
        print(f"Running: {sampler_name} | percentile={p}%")

        steps = [('preprocessor', preprocessor_iterative)]

        if sampler is not None:
            steps.append(('sampler', sampler))

        steps += [
            ('feature_selection', SelectPercentile(score_func=f_classif, percentile=p)),
            ('classifier', rf)
        ]

        imbpipeline = ImbPipeline(steps=steps)


# In[70]:


imbpipeline.fit(X_train,y_train)


# In[71]:


results_percentile_iterative = []

for p in [30, 50, 70, 90, 95]:
    for sampler_name, sampler in sampling_strategies.items():
        for i in range(len(list_model)):
            model = list_model[i]
            model_name = nama_model[i]

            steps = [('preprocessor', preprocessor_iterative)]

            if sampler is not None:
                steps.append(('sampler', sampler))

            steps += [
                ('feature_selection', SelectPercentile(score_func=f_classif, percentile=p)),
                ('classifier', model)
            ]

            pipeline = ImbPipeline(steps=steps)

            # Evaluasi
            result = cross_validate(pipeline, X_train, y_train,
                                    scoring=f2,
                                    cv=skfold,
                                    return_train_score=False)

            results_percentile_iterative.append({
                'Percentile': p,
                'Sampler': sampler_name,
                'Model': model_name,
                'Avg F2 Score': result['test_score'].mean(),
                'F2 Score Std': result['test_score'].std(),
                'Avg Fit Time': result['fit_time'].mean(),
                'Avg Score Time': result['score_time'].mean()
            })


df_percentile_iterative = pd.DataFrame(results_percentile_iterative)
df_percentile_iterative = df_percentile_iterative.sort_values(by='Avg F2 Score', ascending=False).reset_index(drop=True)
display(df_percentile_iterative.head(50))


# ### Top 3 Model Performance with SelectPercentile + Iterative Imputer
# 
# | Percentile | Sampler           | Model          | Avg F2 Score | F2 Score Std | Avg Fit Time | Avg Score Time |
# |------------|-------------------|----------------|--------------|---------------|----------------|------------------|
# | 90         | RandomOverSampler | XGBoost        | **0.8412**   | 0.0202        | 0.4557         | 0.0191           |
# | 90         | RandomOverSampler | LGBMClassifier | **0.8375**   | 0.0122        | 0.5061         | 0.0173           |
# | 95         | RandomOverSampler | XGBoost        | **0.8372**   | 0.0178        | 0.4223         | 0.0121           |
#   
# #### Insights
# 
# - Using SelectPercentile with all features (90%) and RandomOverSampler, XGBoost outperforms all with the highest and most stable F2 score (0.8412).
# - LGBMClassifier follows closely with 0.8375, though with slightly more long in training time.

# ### Summary Feature Selection 

# In[72]:


top3_selectFromModel_median = df_result.head(3)
top3_selectKBest_median = df_kbest.head(3)
top3_selectPercentile_median = df_percentile.head(3)

combined_all_feature_selection_median = pd.concat([top3_selectFromModel_median, top3_selectKBest_median, 
                                             top3_selectPercentile_median], ignore_index=True)


combined_all_feature_selection_median = combined_all_feature_selection_median.sort_values(
    by='Avg F2 Score', ascending=False
).reset_index(drop=True)


display(Markdown("**Top Models from Each Feature Selection Method (Median Imputer)**"))
display(combined_all_feature_selection_median)



top3_selectFromModel_iterative = df_result_IterativeSelectFromModel.head(3)
top3_selectKBest_iterative = df_kbest_iterative.head(3)
top3_selectPercentile_iterative = df_percentile_iterative.head(3)


combined_all_feature_selections_iterative = pd.concat([top3_selectFromModel_iterative,
                                            top3_selectKBest_iterative, top3_selectPercentile_iterative], ignore_index=True)

combined_all_feature_selections_iterative = combined_all_feature_selections_iterative.sort_values(
    by='Avg F2 Score', ascending=False
).reset_index(drop=True)

display(Markdown("**Top Models from Each Feature Selection Method (Iterative Imputer)**"))
display(combined_all_feature_selections_iterative)


# ### Final Model Selection 
# 
# After extensive experimentation with various models, imputers, feature selection techniques, and sampling strategies, we selected the following configuration as our final model:
# 
# - **Model**: `LGBMClassifier`
# - **Imputer**: `Median Imputer`
# - **Feature Selection**: `SelectKBest` with `k=24` 
# - **Sampling Strategy**: `RandomOverSampler`
# 
# This choice was made based on a holistic evaluation of model performance, stability, and computational efficiency, such as:
# 
# #### Strong Classification Performance
# 
# - The combination of `LGBMClassifier` and `Median Imputer` achieved an **average F2 Score of 0.8514**.
# - We chose **F2 Score** as the main evaluation metric to prioritize **minimizing false negatives**, which is critical in churn prediction to avoid overlooking high-risk customers.
# 
# #### Sufficient Model Stability
# 
# - The **F2 Score standard deviation** for `LGBM + Median Imputer` is **0.0103**, which indicates stable performance across different cross-validation folds.
# 
# #### Superior Computational Efficiency
# 
# - `Median Imputer` is **much faster and simpler** than `Iterative Imputer`, leading to:
#   - **Avg Fit Time**: 0.12s for LGBM (vs 0.42–0.64s for Iterative).
#   - **Avg Score Time**: 0.011s (vs 0.016–0.018s for Iterative).
# - This makes the model pipeline **easier to maintain and deploy**, especially for regular retraining or hyperparameter tuning in production.
# 
# #### Simplicity and Interpretability
# 
# - Using `SelectKBest` with the **top 24 features** ensures the model remains **concise and interpretable** without being overwhelmed by irrelevant variables.
# - LGBM provides easy access to **feature importance**, which supports explainability to stakeholders and domain experts.
# 
# 
# The selected configuration, **LGBMClassifier + Median Imputer + SelectKBest + RandomOverSampler**, delivers the best trade-off between:
# 
# - **High classification performance (F2 Score)**
# - **Model consistency (low standard deviation)**
# - **Computational efficiency (low training and scoring time)**
# - **Simplicity and deployment readiness**
# 
# For these reasons, it was chosen as the **final model** for our churn prediction task.

# ### Hyperparameter Tuning

# In[91]:


sampler_tuning = RandomOverSampler(random_state=79)

pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),  
    ('sampler', sampler_tuning),
    ('feature_selection', SelectKBest(score_func=f_classif)),
    ('classifier', LGBMClassifier())])

param_distributions = [

    # LGBM
    {
        'feature_selection__k': [24],  
        'classifier': [LGBMClassifier(random_state=79)],
        'classifier__n_estimators': randint(100, 400),
        'classifier__max_depth': randint(3, 12),
        'classifier__learning_rate': uniform(0.01, 0.2),
        'classifier__subsample': uniform(0.6, 0.4),
        'classifier__colsample_bytree': uniform(0.6, 0.4),
    }
]

randomcv = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=30,                
    scoring=f2,               
    cv=skfold,                
    n_jobs=-1,
    verbose=2,
    random_state=79
)

start_time = time.time()
randomcv.fit(X_train, y_train)
end_time = time.time()


# In[92]:


print(f"Fitting Time: {end_time - start_time:.2f} seconds")
print("Best Params:")
display(randomcv.best_params_)
print(f"Best F2 Score: {randomcv.best_score_:.4f}")


# In[93]:


randomcv.best_estimator_


# In[94]:


randomcv.best_params_


# In[95]:


pd.DataFrame(randomcv.cv_results_).sort_values('rank_test_score')


# In[97]:


y_pred_test=randomcv.best_estimator_.predict(X_test)
f2_test = fbeta_score(y_test, y_pred_test, beta=2)

df_eval = pd.DataFrame({
    'Data': ['Test'],
    'F2 Score': [f2_test]
})

display(df_eval)


# #### Post-Tuning Performance
# 
# After hyperparameter tuning, the final model achieved an **F2 Score of 91% on the test set**, indicating a significant performance boost from the validation stage. This result confirms that the chosen pipeline generalizes well to unseen data and is highly effective in minimizing false negatives, which is crucial for churn prediction tasks.

# In[98]:


cm = confusion_matrix(y_test, y_pred_test)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Greens')


# #### Insights
# ##### Actual vs Predicted:
# 
# **1. Actual Non-Churn (0)**:
# 
# - Correctly Predicted as Non-Churn (TN): 921
# 
# - Wrongly Predicted as Churn (FP): 15
# 
# **2. Actual Churn (1)**:
# 
# - Correctly Predicted as Churn (TP): 173
# 
# - Wrongly Predicted as Non-Churn (FN): 17
# 
# * This Means:
#     - Model is good at spotting Non-Churn customers (921 correct).
#     - But misses 17 Churn cases (predicted as Non-Churn when they actually left).
#     - 15 False Alarms (predicted as Churn but actually stayed).

# ### Feature Importance 

# In[99]:


best_pipeline = randomcv.best_estimator_

feature_names = best_pipeline.named_steps['preprocessor'].get_feature_names_out()

mask = best_pipeline.named_steps['feature_selection'].get_support()

selected_features = feature_names[mask]

importances = best_pipeline.named_steps['classifier'].feature_importances_

feat_imp_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

display(feat_imp_df)


# In[100]:


feat_imp_df.to_csv("feature_importance.csv", index=False)


# In[101]:


colors = [
    '#082f14',
    '#0d3b1a',
    '#1b5e20',
    '#2e7d32',
    '#388e3c',
    '#43a047',
    '#4caf50',
    '#66bb6a',
    '#81c784',
    '#a5d6a7'  
]

plt.figure(figsize=(10, 6))
sns.barplot(
    data=feat_imp_df.head(len(colors)),  
    y='Feature',
    x='Importance',
    palette=colors
)
plt.title('Top 10 Feature Importance (LGBM + SelectKBest)', fontsize=14)
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


# #### Insights from Feature Importance Analysis:
# **1.Top Influential Features**:
# 
# - `CashbackAmount` (1161) is the most important feature, indicating that customers are highly sensitive to cashback offers. This suggests that financial incentives play a crucial role in customer behavior.
# 
# - `WarehouseToHome` (869) and `OrderAmountHikeFromlastYear` (697) are also significant, highlighting the importance of delivery efficiency and changes in spending habits.
# 
# **2.Customer Loyalty and Engagement**:
# 
# - `Tenure` (631) and `DaySinceLastOrder` (513) reflect customer retention and activity. Longer tenure and recent orders correlate strongly with positive outcomes, emphasizing the need to maintain engagement.
# 
# **3. Behavioral and Demographic Factors**:
# 
# - `SatisfactionScore` (472) and `CityTier` (223) show that customer satisfaction and location impact decisions. Urban customers (higher city tiers) may have different expectations.
# 
# - `Gender_Male` (148) and `MaritalStatus_0` (156) suggest demographic targeting could be refined, though their influence is moderate compared to numerical features.
# 
# **4. Less Impactful Features**:
# 
# - Payment modes, order categories (`PreferredOrderCat_0` at 35), and address counts have minimal impact. These may be deprioritized in model optimization or marketing strategies.

# ## Model Constraint

# In[105]:


X_train.describe()


# In[106]:


pd.set_option('display.max_colwidth', None)


columns = X_train.select_dtypes(exclude='number').columns
values = []
nunique_list = []
for i in columns:
    value = X_train[i].unique()
    nunique = X_train[i].nunique()
    nunique_list.append(nunique)
    values.append(value)

display(
    pd.DataFrame({
    "columns" : columns,
    "values" : values,
    "nunique": nunique_list
})
       )
pd.reset_option('display.max_colwidth')


# #### **Numerical Features – Valid Value Ranges**
# 
# | Feature Name                    | Data Type | Min   | Max     |
# |--------------------------------|-----------|-------|---------|
# | `Tenure`                       | Integer   | 0     | 61      |
# | `WarehouseToHome`              | Integer   | 5     | 126     |
# | `HourSpendOnApp`               | Float     | 0.0   | 5.0     |
# | `NumberOfDeviceRegistered`     | Integer   | 1     | 6       |
# | `OrderAmountHikeFromlastYear`  | Integer   | 11    | 26      |
# | `CouponUsed`                   | Integer   | 0     | 16      |
# | `OrderCount`                   | Integer   | 1     | 16      |
# | `DaySinceLastOrder`            | Integer   | 0     | 46      |
# | `CashbackAmount`               | Float     | 0.00  | 324.99  |
# 
# 
# #### **Categorical Features – Allowed Values**
# 
# | Feature Name            | Categories                                                                 |
# |-------------------------|----------------------------------------------------------------------------|
# | `PreferredLoginDevice`  | `Mobile Phone`, `Computer`                                                 |
# | `CityTier`              | `1`, `2`, `3`                                                  |
# | `PreferredPaymentMode`  | `Debit Card`, `Credit Card`, `E wallet`, `Cash on Delivery`, `UPI`         |
# | `Gender`                | `Female`, `Male`                                                           |
# | `PreferedOrderCat`      | `Mobile Phone`, `Laptop & Accessory`, `Grocery`, `Fashion`, `Others`       |
# | `SatisfactionScore`     | `1`, `2`, `3`, `4`, `5`                                  |
# | `MaritalStatus`         | `Single`, `Married`, `Divorced`                                           |
# | `Complain`              | `0`, `1` *(0 = not complain, 1 = complain)*                             |
# | `CountOfAddress`        | `1–2`, `3`, `4–6`, `7+` 
# 
# 
# This model was developed and trained using historical data with the specific features listed above. **Each feature has a defined valid range (for numerical variables) or a fixed set of categories (for categorical variables)** based on the training dataset.
# 
# Therefore:
# 
# 1. **If future data contains values outside of these ranges** or **new/unseen categories**, then:
#   - The model may **fail to recognize or handle** those inputs properly.
#   - Predictions may become **inaccurate or unreliable**.
# 
# 2. To maintain reliable and consistent performance, it is crucial to:
#   - **Ensure incoming data remains within the defined constraints**.
#   - **Continuously monitor for distribution shifts or emerging values** in production data.
#   - **Retrain or update the model** whenever significant deviations are observed.
# 
# In summary, **this model is not fully general-purpose**, and periodic updates or retraining will be necessary **if new patterns or values emerge in the future** that were not present during training.

# ## Conclusion and Recommendation

# #### Conclusion
# - This project successfully developed a robust and interpretable machine learning pipeline to predict customer churn in an e-commerce platform. By prioritizing the F2 Score, the modeling process emphasized minimizing false negatives, a critical factor for ensuring high-risk customers are not overlooked. The final model, a **LightGBM classifier** combined with **Median Imputer**, **SelectKBest** `k=24` feature selection, and **RandomOverSampler**, demonstrated outstanding performance with a **91% F2 Score** on the test set. This configuration not only delivered high predictive power but also showed stable results across folds and superior computational efficiency, making it suitable for future deployment and retraining.
# - Key drivers of churn include financial incentives, behavioral patterns, and demographic factors. Customers with **low cashback received**, **short tenure**, or who have recently **lodged complaints** are **significantly more likely to churn**. Interestingly, even customers reporting very high satisfaction scores (score of 5) exhibit elevated churn rates—potentially **indicating a mismatch** between satisfaction and loyalty. Additionally, churn is more prevalent among customers from lower-tier cities, highlighting geographic and service-level disparities that may affect churn.
# - Overall, this predictive model equips the **Customer Marketing Team** with the ability to proactively identify and engage high-risk customers. By leveraging both machine learning predictions and feature importance insights, the company can optimize retention campaigns, reduce marketing waste, and improve long-term customer loyalty.

# #### Business Recommendation 
# 
# **1. CashbackAmount**
# - **Insight**: This is the most influential feature in the model. Retained customers consistently receive more cashback compared to those who churn.
# - **Reasoning**: Cashback acts as a financial incentive that strengthens customer retention. Churned customers tend to have shorter tenure and longer time since last order, suggesting they did not experience enough value early in the customer journey.
# - **Recommendation**:
#   - Introduce targeted cashback programs for new users or high-risk customers with low tenure.
#   - Implement tiered cashback schemes to reward loyalty and encourage repeat purchases.
# 
# **2. WarehouseToHome**
# - **Insight**: Customers located farther from the warehouse show a slightly higher churn rate.
# - **Reasoning**: Longer delivery distances may result in slower fulfillment and reduced satisfaction, especially in more remote areas.
# - **Recommendation**:
#   - Offer free shipping or delivery discounts for customers beyond a certain distance threshold.
#   - Optimize logistics for Tier 3 cities or remote locations to reduce churn caused by fulfillment delays.
# 
# **3. Satisfaction Score**
# - **Insight**:
# Surprisingly, customers with the highest satisfaction scores (score of 5) churn more frequently than those with lower scores.
# **Reasoning**:
# This counterintuitive pattern may be attributed to:
#   - Survey bias, such as default 5-star submissions without genuine feedback.
#   - Bot-generated reviews or vendor manipulation, where sellers may inflate ratings to appear credible while providing subpar service.
#   - Data quality issues, including mislabeling, outdated records, or inconsistencies between satisfaction data and churn labels.
# - **Recommendation**:
#   - Conduct a qualitative audit on high-satisfaction churners to investigate signs of fraudulent behavior or data entry automation.
#   - Evaluate whether certain vendors exploit the rating system for visibility while causing customer churn.
#   - Reassess the current satisfaction scoring system’s validity and granularity—consider adopting a more robust approach like a multi-question NPS or post-interaction feedback surveys.
# 
# **4. Order Count**
# - **Insight**: Churned customers generally have fewer repeat orders, especially in high-value categories like mobile phones.
# - **Reasoning**: Low order count reflects limited engagement and a higher likelihood of one-time use behavior.
# - **Recommendation**:
#   - Target low-order customers with personalized reminders, post-purchase surveys, or "win-back" offers.
#   - Set thresholds (e.g., <2 orders in 60 days) to activate churn prevention campaigns.
# 
# **5. Complain**
# - **Insight**: Customers who file complaints show a noticeably elevated churn tendency.
# - **Reasoning**: Complaint submission indicates a friction point or dissatisfaction moment that wasn't resolved effectively.
# - **Recommendation**:
#   - Enhance complaint resolution speed and track post-resolution satisfaction.
#   - Audit complaint types to find UI/UX or product-level pain points and fix root causes.
#   - Reward feedback with discount vouchers or service upgrades to rebuild trust.

# #### Model Recommendations
# To ensure long-term model effectiveness and business adaptability, the following recommendations are proposed:
# 
# **1. Regular Monitoring of Model Performance** 
# - Continuously track performance metrics—especially the F2 Score and False Negative Rate—to ensure the model still captures churn risk accurately. High false negatives may lead to lost opportunities in retaining valuable customers.
# 
# **2. Set a periodic model evaluation schedule**
# - If the model’s predictive power significantly drops or business dynamics shift (e.g., new customer behavior trends, product updates), retrain or update the model using the latest data.
# 
# **3. Implement a Feedback Loop**
# - Establish a feedback system where actual churn outcomes are fed back into the system. This allows for incremental learning or model fine-tuning based on real-world results.
# 
# **4. Build a Versioning System**
# - Maintain version control of different model iterations to compare performance over time and ensure traceability.

# ### Save Model

# In[107]:


model_filename = "final_tuned_lightgbm_ros_selectkbest.pkl"

with open(model_filename, "wb") as file:
    pickle.dump(randomcv.best_estimator_, file)

print(f"Model saved to {model_filename}")


# In[ ]:





# In[ ]:




