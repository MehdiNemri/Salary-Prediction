import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from salary_prediction import categorize_job_title, group_education

class TestSalaryPrediction(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.sample_data = pd.DataFrame({
            'Job Title': ['Software Engineer', 'Data Scientist', 'Manager'],
            'Education Level': ['Bachelor', 'Master', 'PhD'],
            'Age': [25, 30, 40],
            'Years of Experience': [2, 5, 15],
            'Gender': ['Male', 'Female', 'Other'],
            'Salary': [60000, 120000, 200000]
        })

    def test_categorize_job_title(self):
        # Test the job title categorization function
        self.assertEqual(categorize_job_title('Software Engineer'), 'Software/Developer')
        self.assertEqual(categorize_job_title('HR Manager'), 'HR/Human Resources')
        self.assertEqual(categorize_job_title('Random Title'), 'Other')

    def test_group_education(self):
        # Test the education grouping function
        self.assertEqual(group_education('Bachelor degree'), 'Bachelor')
        self.assertEqual(group_education('PhD'), 'PhD')
        self.assertEqual(group_education('Unknown'), 'Other')

    def test_missing_values_handling(self):
        # Ensure missing values are handled correctly
        data_with_nan = self.sample_data.copy()
        data_with_nan.loc[0, 'Salary'] = np.nan
        cleaned_data = data_with_nan.dropna()
        self.assertEqual(cleaned_data.shape[0], self.sample_data.shape[0] - 1)

    def test_model_prediction(self):
        # Test Decision Tree and Random Forest regressors
        X = self.sample_data[['Age', 'Years of Experience']]
        y = self.sample_data['Salary']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Decision Tree
        dtree = DecisionTreeRegressor()
        dtree.fit(X_train, y_train)
        predictions = dtree.predict(X_test)
        self.assertEqual(len(predictions), len(X_test))

        # Random Forest
        rforest = RandomForestRegressor()
        rforest.fit(X_train, y_train)
        predictions = rforest.predict(X_test)
        self.assertEqual(len(predictions), len(X_test))

if __name__ == '__main__':
    unittest.main()
