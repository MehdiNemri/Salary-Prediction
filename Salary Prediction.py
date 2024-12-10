#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries
# 

# In[1]:


# Import necessary libraries for data manipulation, analysis, and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Loading the dataset
# Reading the CSV file containing salary data based on country and race demographics

try:
    df = pd.read_csv('Salary_Data_Based_country_and_race.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: The file 'Salary_Data_Based_country_and_race.csv' was not found. Please check the file path.")

# Display the first few rows of the dataset to get an overview of its structure and content
df.head()


# ## Data Preprocessing

# In[3]:


# Checking the shape of the dataset
# This will output the number of rows (samples) and columns (features) in the dataset
num_rows, num_columns = df.shape
print(f"The dataset contains {num_rows} rows and {num_columns} columns.")


# In[4]:


# Checking for missing values in each column of the dataset

missing_values = df.isnull().sum()

# Display only columns with missing values, if any
missing_columns = missing_values[missing_values > 0]

if not missing_columns.empty:
    print("Columns with missing values:\n", missing_columns)
else:
    print("No missing values found in the dataset.")


# Given that the number of rows with missing values is minimal compared to the total dataset size, we can drop these rows to maintain data integrity without significantly impacting the dataset.

# In[5]:


# Dropping rows with missing values
# This step removes any rows that contain null values to ensure a clean dataset for analysis

initial_row_count = df.shape[0]  # Store initial row count for comparison
df.dropna(axis=0, inplace=True)

# Confirm that rows with missing values have been dropped
rows_dropped = initial_row_count - df.shape[0]
print(f"Rows dropped due to missing values: {rows_dropped}")
print(f"New dataset shape: {df.shape}")


# In[6]:


#checking for null values
df.isnull().sum()


# In[7]:


# Dropping the unnecessary column
# The 'Unnamed: 0' column is likely an index column created during data import, so we remove it for cleaner data

if 'Unnamed: 0' in df.columns:
    df.drop(columns='Unnamed: 0', axis=1, inplace=True)
    print("Column 'Unnamed: 0' has been dropped.")
else:
    print("Column 'Unnamed: 0' not found in the dataset.")

# Display the updated column list
print("Current columns in the dataset:", df.columns.tolist())


# Check data type of each column

# In[8]:


df.dtypes


# Check for unique values in each column

# In[9]:


#unique values in each column
df.nunique()


# The job title column contains 191 unique values, making it challenging to analyze individually. Therefore, we can group similar job titles into broader job categories to simplify the analysis.

# #### Grouping Job Titles

# In[12]:


# Displaying unique job titles in the dataset
# This helps in understanding the variety of job titles before grouping them into broader categories

unique_job_titles = df['Job Title'].unique()
print(f"Number of unique job titles: {len(unique_job_titles)}")
print("Sample of unique job titles:", unique_job_titles[:20])  # Display the first 20 unique job titles as a sample


# In[13]:


# Function to categorize job titles into broader domains for simplified analysis
def categorize_job_title(job_title):
    # Convert job title to lowercase for case-insensitive matching
    job_title = str(job_title).lower()
    
    # Categorize based on keywords in job title
    if 'software' in job_title or 'developer' in job_title:
        return 'Software/Developer'
    elif 'data' in job_title or 'analyst' in job_title or 'scientist' in job_title:
        return 'Data Analyst/Scientist'
    elif 'manager' in job_title or 'director' in job_title or 'vp' in job_title:
        return 'Manager/Director/VP'
    elif 'sales' in job_title or 'representative' in job_title:
        return 'Sales'
    elif 'marketing' in job_title or 'social media' in job_title:
        return 'Marketing/Social Media'
    elif 'product' in job_title or 'designer' in job_title:
        return 'Product/Designer'
    elif 'hr' in job_title or 'human resources' in job_title:
        return 'HR/Human Resources'
    elif 'financial' in job_title or 'accountant' in job_title:
        return 'Financial/Accountant'
    elif 'project manager' in job_title:
        return 'Project Manager'
    elif 'it' in job_title or 'support' in job_title:
        return 'IT/Technical Support'
    elif 'operations' in job_title or 'supply chain' in job_title:
        return 'Operations/Supply Chain'
    elif 'customer service' in job_title or 'receptionist' in job_title:
        return 'Customer Service/Receptionist'
    else:
        return 'Other'

# Apply the categorization function to the 'Job Title' column in the dataset
df['Job Title'] = df['Job Title'].apply(categorize_job_title)

# Display the updated job title categories
print("Job title categories after grouping:")
print(df['Job Title'].value_counts())


# In[14]:


df['Education Level'].unique()


# There are variations in education levels (e.g., "Bachelor" and "Bachelor degree") that represent the same level.  We'll replace these with a consistent label for clearer analysis.

# #### Grouping Education Level

# In[15]:


# Function to group education levels into standardized categories for consistency
def group_education(education):
    # Convert to lowercase for case-insensitive matching
    education = str(education).lower()
    
    # Standardize education levels based on common keywords
    if 'high school' in education:
        return 'High School'
    elif 'bachelor' in education:  # Handles both "Bachelor" and "Bachelor's"
        return 'Bachelor'
    elif 'master' in education:  # Handles both "Master" and "Master's"
        return 'Master'
    elif 'phd' in education or 'doctorate' in education:
        return 'PhD'
    else:
        return 'Other'  # For any unexpected values
    
# Apply the grouping function to the 'Education Level' column
df['Education Level'] = df['Education Level'].apply(group_education)

# Display the updated education level categories
print("Standardized Education Level values:\n", df['Education Level'].value_counts())
  


# #### Descriptive Statistics

# In[16]:


#descriptive statistics
df.describe()


# In[17]:


df.head()


# ## Exploratory Data Analysis
# 
# In the exploratory data analysis, I will start by examining the dataset to gain a clear understanding of its structure.

# ### Pie chart for Gender

# In[18]:


# Pie chart for gender distribution in the dataset
# This visualization shows the proportion of each gender in the dataset.

plt.figure(figsize=(8, 6))  # Adjusting the figure size for better readability
gender_counts = df['Gender'].value_counts()  # Count values for each gender category

# Plotting the pie chart
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral', 'gold'])
plt.title('Gender Distribution in the Dataset')
plt.show()


# ### Age Distribution

# In[19]:


# Histogram for age distribution in the dataset
# This plot shows the distribution of ages with a KDE overlay for a smoother view of the density.

plt.figure(figsize=(10, 6))  # Set figure size for better readability
sns.histplot(data=df, x='Age', bins=20, kde=True, color='skyblue')  # KDE adds a smooth density estimate
plt.title('Age Distribution in the Dataset')
plt.xlabel('Age')  # Label for x-axis
plt.ylabel('Frequency')  # Label for y-axis
plt.show()


# The majority of employees fall within the 25–35 age range, indicating a predominantly young and dynamic workforce. Only a small portion of employees are over 55 years old, highlighting a minimal representation of older individuals in the dataset.

# ### Education Level

# In[ ]:


sns.countplot(x = 'Education Level', data = df, palette='Set1')
plt.xticks(rotation=90)


# Most of the employees have a Bachelor's degree followed by Master's degree and Doctoral degree. The least number of employees have a High School education. From the graph it is clear that most of the employees started working after graduation, few of them started working after post graduation and very few of them have gone for doctorate. The least number of employees have started working after high school education.

# ### Job Title

# In[ ]:


sns.countplot(x='Job Title', data = df)
plt.xticks(rotation=90)


# This graph helps us to breakdown the data of job title in a simpler form. From the graph, it is clear that majority of the employees have job titles - Software Developer, Data Analyst/Scientist or Manager/Director/Vp. Few amount of employees have job titles such as sales, marketing/social media, HR, Product Designer and Customer Service. Very few of the eomployees work as a Financial/accountant or operation/supply management.
# 
# From this I build a hypothesis that the job titles such as Software Developer, Data Analyst/Scientist and Manager/Director are in more demand as compared to other job titles. It also means that job titles like Financial/accountant or operation/supply management and Customer Service are in less demand and paid comparatively less.

# ### Years of Experience

# In[ ]:


sns.histplot(x = 'Years of Experience', data = df,kde=True)


# Most of the employees in the dataset havr experience of 0-7 years in the respective domains in which particularly majority of them have experience between less than 5 years. Moreover the number of employees in the dataset decreases with increasing number of years of experience.

# ### Country

# In[ ]:


sns.countplot(x='Country', data=df)
plt.xticks(rotation=90)


# The number of employees from the above 5 countries is nearly same, with a little more in USA.

# ### Racial Distribution

# In[ ]:


sns.countplot(x='Race', data=df)
plt.xticks(rotation=90)


# This graph help us to know about the racial distribution in the dataset. From the graph, it is clear that most of the employees are either White or Asian, followed by Korean, Chinese, Australian and Black. Number of employees from Welsh, African American, Mixed and Hispanic race are less as compared to other groups.

# From all the above plots and graphs, we can a understanding about the data we are dealing with, its distribution and quantity as well. Now I am gonna explore the realtion of these independent variables with the target Variable i.e. Salary.

# ### Age and Salary

# In[ ]:


sns.scatterplot(x = 'Age', y='Salary', data=df)
plt.title('Age vs Salary')


# In this scatter plot we see a trend that the salary of the person increases with increse in the age, which is obvious because of promotion and apprisals. However upon closer observation we can find that similar age have multiple salaries, which means there are other factors which decides the salary.

# ### Gender and Salary

# In[ ]:


fig, ax = plt.subplots(1,2, figsize = (15, 5))
sns.boxplot(x = 'Gender', y='Salary', data = df, ax =ax[0]).set_title('Gender vs Salary')
sns.violinplot(x = 'Gender', y='Salary', data = df, ax =ax[1]).set_title('Gender vs Salary')


# The boxplot and violinplot describes the salary distribution among the three genders. In the boxplot the employees from Other gender has quite high salary as compared to Makes and Females. The other gender employees have a median salary above 150000, followed by males with median salary near 107500 and females with median salary near 100000. The voilin plot visualizes the distribution of salary with respect to the gender, where most of the Other gender employees have salary above 150000. In makes this distribution is concentrated between 50000 and 10000 as well as near 200000. In case of females, there salary distribution is quite spread as compared to other genders with most near 50000.

# ### Education Level and Salary

# In[ ]:


fig,ax = plt.subplots(1,2,figsize=(15,6))
sns.boxplot(x = 'Education Level', y = 'Salary', data = df, ax=ax[0]).set_title('Education Level vs Salary')
sns.violinplot(x = 'Education Level', y = 'Salary', data = df, ax=ax[1]).set_title('Education Level vs Salary')


# The boxplot and violinplot shows the distribution of salary based on the employees education level. The median salary for the Phd holders is highest followed by Masters and bachelors degreee holders, with employees with no degree having the lowest median salary. In the violinplot the phd scholars have distribution near 200000, whereas Masters degree holders have a very sleak distribution where the salary distribution is spread from 100k to 150k, The Bachelors degree holders have a salary distribution near 50000 whereas the employees with no degree have a salary distribution near 40k-45k.
# 
# From these graph, I assume that the employees with higher education level have higher salary than the employees with lower education level.

# ### Job Title and Salary

# In[ ]:


sns.barplot(x = 'Job Title', y = 'Salary', data = df, palette = 'Set2')
plt.xticks(rotation = 90)


# This graph falsifies my previous hypothesis regarding the demand and paywith respect to job titles. In this graph, 'Other' category job titles have higher salary than those titles which assumed to be in high demand and pay. In contrast to previous Job title graph, this graph shows that there is no relation between the job title distribution and salary. The job titles which gave high salary are found to be less in number.
# 
# However the hypothesis is true about the Job titles such as Software Developer, Data analyst/scuentust and Manager/Director/VP. These job titles are found to be in high demand and pay. But in contrast to that the job titles such as Operation/Supply chain, HR, Financial/Accountant and Marketing/Social Media are found have much more salary as assumed.

# ### Experience and Salary

# In[ ]:


sns.scatterplot(x= 'Years of Experience', y  = 'Salary', data = df).set_title('Years of Experience vs Salary')


# From this scaaterplot, it is clear that on the whole, the salary of the employees is increasing with the years of experience. However, on closer look we can see that similar experience have different salaries. This is because the salary is also dependent on other factors like job title, age, gender education level as discussed earlier.

# ### Country and Salary

# In[ ]:


fig,ax = plt.subplots(1,2,figsize=(15,6))
sns.boxplot(x = 'Country', y = 'Salary', data = df, ax=ax[0])
sns.violinplot(x = 'Country', y = 'Salary', data = df, ax=ax[1])


# Both the boxplot and violinplot shows very similar insight about the salary across all the countiries even in the violinplot distribution. However, there is very small variation in median salary in USA, which is slighlty less as compared to other countries.

# Since, the we cannot get much information about the salary with respect to the countries. So, I will plot the job title vs salary graph for each country, so that we can get a overview of job title vs salary for each country.

# In[ ]:


fig,ax = plt.subplots(3,2,figsize=(20,20))
plt.subplots_adjust(hspace=0.5)
sns.boxplot(x = 'Job Title', y = 'Salary', data = df[df['Country'] == 'USA'], ax = ax[0,0]).set_title('USA')
ax[0,0].tick_params(axis='x', rotation=90)
sns.boxplot(x = 'Job Title', y = 'Salary', data = df[df['Country'] == 'UK'], ax = ax[0,1]).set_title('UK')
ax[0,1].tick_params(axis='x', rotation=90)
sns.boxplot(x = 'Job Title', y = 'Salary', data = df[df['Country'] == 'Canada'], ax = ax[1,0]).set_title('Canada')
ax[1,0].tick_params(axis='x', rotation=90)
sns.boxplot(x = 'Job Title', y = 'Salary', data = df[df['Country'] == 'Australia'], ax = ax[1,1]).set_title('Australia')
ax[1,1].tick_params(axis='x', rotation=90)
sns.boxplot(x = 'Job Title', y = 'Salary', data = df[df['Country'] == 'China'], ax = ax[2,0]).set_title('China')
ax[2,0].tick_params(axis='x', rotation=90)
sns.boxplot(x = 'Job Title', y = 'Salary', data = df, ax = ax[2,1]).set_title('All Countries')
ax[2,1].tick_params(axis='x', rotation=90)


# After observing all these plots, I conclude that the Job Titles such as Softwarre Developer, Manager/Director/VP and Data Analyst/Scientist hare in high demand as well as receive much higer salary than other job titles, excluding the Job Titles that come under 'Other' category. The job titles such as Operation/Supply Chain, Customer Service/Receptionist, Product Designer and sales are in low demand and have low salary.

# ### Race and Salary

# In[ ]:


fig,ax = plt.subplots(1,2,figsize=(15,6))
sns.boxplot(x = 'Race', y = 'Salary', data = df, ax = ax[0])
ax[0].tick_params(axis='x', rotation=90)
sns.violinplot(x = 'Race', y ='Salary', data = df, ax = ax[1])
ax[1].tick_params(axis='x', rotation=90)


# The employees from the races - Australian, Mixed, Blacks and White have the highest median salary, followed by Asian, Korean and Chinese with lowest median salary in employees from hispanic race. Looking at the violinplot the salary distribution is more concentrated after 150k in white, australian, black and mixed race. Whereas the hispanic has more concentration near 75k

# ## Data Preprocessing 2

# ### Label encoding to categorical features

# In[ ]:


from sklearn.preprocessing import LabelEncoder
features = ['Gender','Country','Education Level','Job Title', 'Race']
le = LabelEncoder()
for feature in features:
    le.fit(df[feature].unique())
    df[feature] = le.transform(df[feature])
    print(feature, df[feature].unique())


# ### Normalization   

# In[ ]:


#normalizing the continuous variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['Age', 'Years of Experience', 'Salary']] = scaler.fit_transform(df[['Age', 'Years of Experience', 'Salary']])


# In[ ]:


df.head()


# ## Coorelation Matrix Heatmap

# In[ ]:


#coorelation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),annot=True, cmap='coolwarm')


# In this coorelation matrix, there are three major coorealtions.
# - Salary and Age
# - Salary and Years of Experience
# - Years of Experience and Age
# 
# The coorelation salary with age and years of experience is already explored in the above plots. The coorelation between the years of experience and age is obvious as the person ages the experience will be more.

# ## Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Salary', axis=1), df['Salary'], test_size=0.2, random_state=42)


# ## Salary Prediction

# I will be using the following models:
# - Decision Tree Regressor
# - Random Forest Regressor

# ### Decision Tree Regressor

# In[ ]:


from sklearn.tree import DecisionTreeRegressor

#createing the decision tree gressor object
dtree = DecisionTreeRegressor()


# #### Hypertuning the model

# In[ ]:


from sklearn.model_selection import GridSearchCV

#defining the parameters for the grid search
parameters = {'max_depth' :[2,4,6,8,10],
              'min_samples_split' :[2,4,6,8],
              'min_samples_leaf' :[2,4,6,8],
              'max_features' :['auto','sqrt','log2'],
              'random_state' :[0,42]}
#creating the grid search object
grid_search = GridSearchCV(dtree,parameters,cv=5,scoring='neg_mean_squared_error',n_jobs=-1)

#fit the grid search object to the training data
grid_search.fit(X_train,y_train)

#print the best parameters
print(grid_search.best_params_)


# Building the model on best parameters

# In[ ]:


dtree = DecisionTreeRegressor(max_depth = 10, max_features = 'auto', min_samples_leaf = 2, min_samples_split = 8, random_state = 42)
dtree


# In[ ]:


#fitting the training data
dtree.fit(X_train,y_train)


# In[ ]:


#training accuracy
dtree.score(X_train, y_train)


# In[ ]:


#predicting the salary of an employee 
d_pred = dtree.predict(X_test)


# ## Evaluating the Decision Tree Regressor Model

# In[ ]:


dft = pd.DataFrame({'Actual': y_test, 'Predicted': d_pred})
dft.reset_index(drop=True, inplace=True)
dft.head(10)


# In[ ]:


ax = sns.distplot(dft['Actual'], color = 'blue', hist = False, kde = True, kde_kws = {'linewidth': 3}, label = 'Actual')
sns.distplot(  dft['Predicted'], color = 'red', ax=ax, hist = False, kde = True, kde_kws = {'linewidth': 3}, label = 'Predicted')


# The blue shows the distribution count for actual values and the red line shows the distribution count for predicted values. The predicted values are close to the actual values and ther curve coincides with the actual values curve. This shows that the model is a good fit.

# In[ ]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print("R2 Score: ", r2_score(y_test, d_pred))
print("Mean Squared Error: ", mean_squared_error(y_test, d_pred))
print("Mean Absolute Error: ", mean_absolute_error(y_test, d_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, d_pred)))


# ### Random Forest Regressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
#creating random forest regressor object
rfg = RandomForestRegressor()


# In[ ]:


#trainig the model
rfg.fit(X_train, y_train)


# In[ ]:


#training accuracy
rfg.score(X_train, y_train)


# In[ ]:


#predicitng salary of the employee
r_pred = rfg.predict(X_test)


# ## Evaluating Random Forest Regressor Model

# In[ ]:


dfr = pd.DataFrame({'Actual': y_test, 'Predicted': r_pred})
dfr.reset_index(drop=True, inplace=True)
dfr.head(10)


# In[ ]:


ax = sns.distplot(dft['Actual'], color = 'blue', hist = False, kde = True, kde_kws = {'linewidth': 3}, label = 'Actual')
sns.distplot(  dft['Predicted'], color = 'red', ax=ax, hist = False, kde = True, kde_kws = {'linewidth': 3}, label = 'Predicted')


# The blue shows the distribution count for actual values and the red line shows the distribution count for predicted values. The predicted values are close to the actual values and ther curve coincides with the actual values curve. This shows that the model is a good fit.

# In[ ]:


print("R2 Score: ", r2_score(y_test, r_pred))
print("Mean Squared Error: ", mean_squared_error(y_test, r_pred))
print("Mean Absolute Error: ", mean_absolute_error(y_test, r_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, r_pred)))


# ## Conclusion
# 
# From the exploratory data analysis, I have concluded that the salary of the employees is dependent upon the following factors:
# 1. **Years of Experience**
# 2. **Job Title**
# 3. **Education Level**
# 
# Employees with greater years of experience, having job title such as Data analyst/scientist, Software Developer or Director/Manager/VP and having a Master's or Doctoral degree are more likely to have a higher salary.
# 
# Coming to the machine learning models, I have used regressor models - Decision Tree Regressor and Random Forest Regressor for predicting the salary. The Random Forest Regressor has performed well with the accuracy of 94.6%
