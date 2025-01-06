#!/usr/bin/env python
# coding: utf-8

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from logging_config import setup_logger
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Setting up the logger
logger = setup_logger()

try:
    # Load dataset
    logger.info("Attempting to load dataset...")
    df = pd.read_csv('/data/Salary_Data_Based_country_and_race.csv')
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

# Grouping Education Levels
def group_education(education):
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
# (Visualizations code goes here)

# Save preprocessed data
df.to_csv('/data/processed_data.csv', index=False)
logger.info("Preprocessed data saved successfully.")
