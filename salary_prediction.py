#!/usr/bin/env python
# coding: utf-8

# Importing necessary libraries for data manipulation, analysis, visualization, and logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from logging_config import setup_logger
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Setting up the logger
logger = setup_logger()

try:
    # Load dataset
    logger.info("Attempting to load dataset...")
    df = pd.read_csv('Salary_Data_Based_country_and_race.csv')
    logger.info("Dataset loaded successfully!")
except FileNotFoundError:
    logger.error("The file 'Salary_Data_Based_country_and_race.csv' was not found. Please check the file path.")
    exit()

# Dataset overview
logger.info("Displaying the first few rows of the dataset:")
print(df.head())

num_rows, num_columns = df.shape
logger.info(f"The dataset contains {num_rows} rows and {num_columns} columns.")

# Checking for missing values
missing_values = df.isnull().sum()
missing_columns = missing_values[missing_values > 0]

if not missing_columns.empty:
    logger.warning(f"Columns with missing values:\n{missing_columns}")
else:
    logger.info("No missing values found in the dataset.")

# Dropping rows with missing values
initial_row_count = df.shape[0]
df.dropna(axis=0, inplace=True)
rows_dropped = initial_row_count - df.shape[0]
logger.info(f"Rows dropped due to missing values: {rows_dropped}")
logger.info(f"New dataset shape: {df.shape}")

# Display column names
logger.info("Displaying the columns of the dataset:")
print(df.columns)

# Display data types of columns
logger.info("Displaying data types of each column:")
print(df.dtypes)

# Dropping unnecessary columns
if 'Unnamed: 0' in df.columns:
    df.drop(columns='Unnamed: 0', inplace=True)
    logger.info("Dropped 'Unnamed: 0' column.")
else:
    logger.info("'Unnamed: 0' column not found in the dataset.")

# Grouping Job Titles
def categorize_job_title(job_title):
    """
    Categorize job titles into broader domains for simplified analysis.

    Args:
        job_title (str): The job title to categorize.

    Returns:
        str: A string representing the broader category of the job title.

    Examples:
        >>> categorize_job_title('Software Developer')
        'Software/Developer'

        >>> categorize_job_title('Data Scientist')
        'Data Analyst/Scientist'
    """
    job_title = str(job_title).lower()
    if 'software' in job_title or 'developer' in job_title:
        return 'Software/Developer'
    elif 'data' in job_title or 'analyst' in job_title or 'scientist' in job_title:
        return 'Data Analyst/Scientist'
    elif 'manager' in job_title or 'director' in job_title or 'vp' in job_title:
        return 'Manager/Director/VP'
    elif 'sales' in job_title:
        return 'Sales'
    elif 'marketing' in job_title:
        return 'Marketing/Social Media'
    elif 'product' in job_title:
        return 'Product/Designer'
    elif 'hr' in job_title:
        return 'HR/Human Resources'
    elif 'financial' in job_title:
        return 'Financial/Accountant'
    elif 'project manager' in job_title:
        return 'Project Manager'
    elif 'it' in job_title or 'support' in job_title:
        return 'IT/Technical Support'
    elif 'operations' in job_title:
        return 'Operations/Supply Chain'
    elif 'customer service' in job_title:
        return 'Customer Service/Receptionist'
    else:
        return 'Other'

df['Job Title'] = df['Job Title'].apply(categorize_job_title)
logger.info("Job titles have been categorized.")

# Display unique job titles
logger.info("Displaying unique job titles after categorization:")
print(df['Job Title'].value_counts())

# Grouping Education Levels
def group_education(education):
    """
    Group education levels into broader categories for consistency.

    Args:
        education (str): The education level to group.

    Returns:
        str: A string representing the broader category of the education level.

    Examples:
        >>> group_education('Bachelor degree')
        'Bachelor'

        >>> group_education('PhD in Computer Science')
        'PhD'
    """
    education = str(education).lower()
    if 'high school' in education:
        return 'High School'
    elif 'bachelor' in education:
        return 'Bachelor'
    elif 'master' in education:
        return 'Master'
    elif 'phd' in education or 'doctorate' in education:
        return 'PhD'
    else:
        return 'Other'

df['Education Level'] = df['Education Level'].apply(group_education)
logger.info("Education levels have been standardized.")

# Encoding categorical features
logger.info("Encoding categorical features...")
features = ['Gender', 'Country', 'Education Level', 'Job Title', 'Race']
le = LabelEncoder()
for feature in features:
    df[feature] = le.fit_transform(df[feature])

# Normalizing continuous variables
logger.info("Normalizing continuous variables...")
scaler = StandardScaler()
df[['Age', 'Years of Experience', 'Salary']] = scaler.fit_transform(df[['Age', 'Years of Experience', 'Salary']])

# Visualizations
logger.info("Displaying visualizations...")

# Gender Distribution
plt.figure(figsize=(8, 6))
gender_counts = df['Gender'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral', 'gold'])
plt.title('Gender Distribution in the Dataset')
plt.show()

# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', bins=20, kde=True, color='skyblue')
plt.title('Age Distribution in the Dataset')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Education Level Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Education Level', data=df, palette='Set2')
plt.title('Education Level Distribution')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Job Title and Salary
plt.figure(figsize=(12, 6))
sns.barplot(x='Job Title', y='Salary', data=df, palette='Set3')
plt.title('Job Title vs Salary')
plt.xlabel('Job Title')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.show()

# Train-Test Split
logger.info("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(df.drop('Salary', axis=1), df['Salary'], test_size=0.2, random_state=42)

# Decision Tree Regressor
logger.info("Training Decision Tree Regressor...")
dtree = DecisionTreeRegressor(max_depth=10, min_samples_split=8, min_samples_leaf=2, random_state=42)
dtree.fit(X_train, y_train)
logger.info(f"Decision Tree Training Accuracy: {dtree.score(X_train, y_train):.2f}")

# Evaluate Decision Tree
d_pred = dtree.predict(X_test)
logger.info(f"Decision Tree R2 Score: {r2_score(y_test, d_pred):.2f}")
logger.info(f"Decision Tree RMSE: {np.sqrt(mean_squared_error(y_test, d_pred)):.2f}")

# Random Forest Regressor
logger.info("Training Random Forest Regressor...")
rfg = RandomForestRegressor(n_estimators=100, random_state=42)
rfg.fit(X_train, y_train)
logger.info(f"Random Forest Training Accuracy: {rfg.score(X_train, y_train):.2f}")

# Evaluate Random Forest
r_pred = rfg.predict(X_test)
logger.info(f"Random Forest R2 Score: {r2_score(y_test, r_pred):.2f}")
logger.info(f"Random Forest RMSE: {np.sqrt(mean_squared_error(y_test, r_pred)):.2f}")

logger.info("Salary Prediction Script Completed Successfully!")
