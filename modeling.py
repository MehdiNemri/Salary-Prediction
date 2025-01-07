#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from logging_config import setup_logger

# Setting up the logger
logger = setup_logger()

# Load preprocessed data
try:
    logger.info("Loading preprocessed data...")
    df = pd.read_csv('/data/processed_data.csv')
    logger.info("Preprocessed data loaded successfully!")
except FileNotFoundError:
    logger.error("The file 'processed_data.csv' was not found.")
    exit()

# Splitting dataset
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

logger.info("Modeling completed successfully.")
