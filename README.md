# Salary-Prediction
Project Overview
This project aims to predict the salaries of individuals from various backgrounds, spanning different countries and races, by leveraging demographic factors such as occupation, age, gender, experience, education level, and more. The dataset is sourced from Kaggle, containing 32,561 records with 15 columns. The analysis focuses on eight key features to predict one primary target: the individual's salary.

# Dataset Summary
This dataset provides an extensive set of salary and demographic details, enriched by experience information. It serves as an excellent foundation for analyzing income patterns in relation to socio-demographic variables. Key demographic factors like age, gender, educational attainment, nationality, and race offer diverse variables for exploration. Researchers can use this dataset to investigate income trends across different demographic groups, shedding light on potential earning disparities and other variations.

The inclusion of experience data, specifically years in the profession, adds a dynamic perspective, allowing for analysis of how earnings evolve with career progression. This feature facilitates a deeper understanding of how demographic traits and professional experience jointly influence income. With such depth, this dataset is well-suited for comprehensive studies on income diversity, helping to uncover the complex factors that affect earning potential in modern work environments.

### Data Dictionary

| Column               | Description                                   |
|----------------------|-----------------------------------------------|
| `Age`                | Age of the individual                         |
| `Country`            | Country of residence                          |
| `Job Title`          | Job title or profession                       |
| `Years of Experience`| Professional experience in years              |
| `Education Level`    | Education level of the individual             |
| `Race`               | Racial background of the individual           |
| `Salary`             | Salary of the individual                      |
| `Unnamed: 0`         | Index                                         |


### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/MehdiNemri/Salary-Prediction.git
   cd Salary-Prediction
2. **Install Dependencies Install necessary Python libraries**
   ```bash
   pip install -r requirements.txt

### Building and Packaging the Project

1. **To create a distributable package, ensure `setuptools` and `wheel` are installed:**

   ```bash
   pip install setuptools wheel
   python setup.py sdist bdist_wheel
This command will generate a .whl file and a .tar.gz file in the dist directory, which can be used for distribution or installation.

### Running the Project

### Data Preprocessing and Model Training

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Salary-Prediction.ipynb

Follow the notebook cells step-by-step. Start with data loading and cleaning, then proceed to model training and evaluation.



