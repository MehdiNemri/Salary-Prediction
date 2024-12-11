import os
import sys
import unittest
from salary_prediction import categorize_job_title, group_education
# Ajoutez le répertoire parent au chemin de Python pour pouvoir importer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



class TestSalaryPrediction(unittest.TestCase):
    def test_categorize_job_title(self):
        test_cases = [
            ('Software Developer', 'Software/Developer'),
            ('Data Scientist', 'Data Analyst/Scientist'),
            ('Sales Manager', 'Manager/Director/VP'),
            ('Marketing Specialist', 'Marketing/Social Media'),
            ('HR Coordinator', 'HR/Human Resources'),
            ('Unknown', 'Other')
        ]

        for job_title, expected_category in test_cases:
            with self.subTest(job_title=job_title):
                self.assertEqual(categorize_job_title(job_title), expected_category)

    def test_group_education(self):
        test_cases = [
            ('Bachelor degree', 'Bachelor'),
            ('Master of Science', 'Master'),
            ('PhD in Computer Science', 'PhD'),
            ('High School Diploma', 'High School'),
            ('Associate Degree', 'Other')
        ]

        for education, expected_group in test_cases:
            with self.subTest(education=education):
                self.assertEqual(group_education(education), expected_group)

if __name__ == '__main__':
    unittest.main()
