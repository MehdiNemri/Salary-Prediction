# Salary Prediction Project: Tutorial

This tutorial provides a step-by-step guide to set up, run, and understand the Salary Prediction project.

## Overview

The Salary Prediction project uses machine learning to predict salaries based on various demographic and professional attributes such as job title, education level, gender, and more.

Key Features:
- Data preprocessing, cleaning, and visualization.
- Machine learning models: Decision Tree and Random Forest.
- Customizable logging for debugging and analysis.

---

## Prerequisites

Before you begin, ensure the following:
- **Python Version**: Python 3.8 or higher.
- **Required Libraries**:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - logging

Install these dependencies using:
```bash
pip install -r requirements.txt


## Getting Started
Start by cloning the project repository:

    ```bash
   git clone https://github.com/MehdiNemri/Salary-Prediction.git
   cd Salary-Prediction
1. Run the Salary Prediction Script : Execute the script to preprocess data, generate visualizations, and train models:
    ```bash
   python3 salary_prediction.py
   
   Outputs:

     Preprocessed data.
     Visualizations displayed in the browser.
     Model performance metrics logged to the console and saved to logs/.	
2. Customize Parameters : Modify model hyperparameters in salary_prediction.py to experiment with different settings:
    ```bash 
   dtree = DecisionTreeRegressor(max_depth=10, random_state=42)
    ```bash 
   rfg = RandomForestRegressor(n_estimators=100, random_state=42)

3. Unit Testing : To verify the core functions of the project:
    ```` bash
    python3 -m pytest tests

## Generated Visualizations : 
When you run the project, the following visualizations are displayed:

	* Gender Distribution: Pie chart of gender proportions.
	* Age Distribution: Histogram with KDE for age.
	* Education Levels: Bar chart of education levels.
	* Job Title vs. Salary: Bar chart of average salaries by job title.
## Logging
All logs are saved to the logs/ directory with timestamps for easy debugging:

	* INFO logs: General information such as dataset loading and preprocessing.
	* DEBUG logs: Detailed metrics for model performance.
	* WARNING/ERROR logs: Alerts for issues such as missing data.
