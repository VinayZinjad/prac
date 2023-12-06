
## Slip 1
Write a Python program build Decision Tree Classifier using Scikit-learn package for diabetes data set (download database from https://www.kaggle.com/uciml/pima-indians-diabetes-database
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
dataset_path = "./diabetes.csv"

df = pd.read_csv(dataset_path)

# Display the first few rows of the dataset
print(df.head())

# Preprocess the data
X = df.drop('Outcome', axis=1)  # Features
y = df['Outcome']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_rep)

```

## Slip 2
```txt
. Consider following dataset weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny' ,'Sunny','Rainy','Sunny','Overcast','Overcast','Rainy'] temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild ','Mi ld','Mild','Hot','Mild'] play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Ye s','Y es','No']. Use Naïve Bayes algorithm to predict[ 0:Overcast, 2:Mild] tuple belongs to which class whether to play the sports or not
```
```python
# %%
# Assigning features and label variables
weather = ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny',
           'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy']
temp = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
        'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild']
play = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
        'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']

# %%
# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
wheather_encoded=le.fit_transform(weather)
print ("Weather:",wheather_encoded)

# Converting string labels into numbers
temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)
print ("Temp:",temp_encoded)
print ("Play:",label)
#Combinig weather and temp into single listof tuples
features=list(zip(wheather_encoded,temp_encoded))
print (features)
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
#Create a Gaussian Classifier
model = GaussianNB()
# Train the model using the training sets
model.fit(features,label)
#Predict Output
predicted= model.predict([[0,2]]) # 0:Overcast, 2:Mild
print("Predicted Value:", predicted)
```


## Slip 3

Write a python program to implement multiple Linear Regression model for a car dataset. Dataset can be downloaded from: https://www.w3schools.com/python/python_ml_multiple_regression.asp

```python
# Multiple Regression
# Multiple regression is like linear regression, but with more than one independent value,
# meaning that we try to predict a value based on two or more variables.
import pandas
from sklearn import linear_model
# Car, Model, Volume, Weight, CO2
df = pandas.read_csv("../input/linear-regression-dataset/cars.csv")
# print(df) # 0, Toyota, Aygo, 1000, 790, 99
X = df[['Weight', 'Volume']]
y = df['CO2']
regr = linear_model.LinearRegression()
# Tip: It is common to name the list of independent values with a upper case X,
# and the list of dependent values with a lower case y.
regr.fit(X, y)
# predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300ccm:
predictedCO2 = regr.predict([[2300, 1300]])
print("predictedCO2 [weight=2300kg, volume=1300ccm]:")
print(predictedCO2) # [107.2087328]
# Coefficient
# The coefficient is a factor that describes the relationship with an unknown variable.
print("Coefficient [weight, volume]")
print(regr.coef_) # [0.00755095 0.00780526]
df
predictedCO2 = regr.predict([[3300, 1300]])
print("predictedCO2 [weight=3300kg, volume=1300ccm]")
print(predictedCO2)
```


## Slip 4
Q.Write a python program to implement k-means algorithm to build prediction model (Use Credit Card Dataset CC GENERAL.csv Download from kaggle.com)

My 
```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset  = pd.read_csv("./CC GENERAL.csv")
X = dataset.iloc[:,1:].values
X

from sklearn.impute import SimpleImputer

# Assuming your dataset is stored in the variable X
# Create a SimpleImputer instance with strategy 'most_frequent'
imputer = SimpleImputer(strategy="most_frequent")

# Fit and transform the data
X = imputer.fit_transform(X)


# Applying Feature Scalling with StandardScaler
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# For Finding Optimal Number of Cluster use Elbow Method 
from sklearn.cluster import KMeans
wcss = []
for i in range(1,18):
 kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state
=0).fit(X)
 wcss.append(kmeans.inertia_)
 
plt.plot(range(1,18), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Cluster')
plt.ylabel('WCSS')
plt.show()


# Apply K-Means Again With Optimal Number of Cluster that we got from Elbow method i.e. 8
kmeans = KMeans(n_clusters=8, init='k-means++', max_iter=300, n_init=10, random_state=
0)
y_kmeans = kmeans.fit_predict(X)
# Finally Append new Column i.e Cluster to Actual Dataset
dataset['Cluster'] = y_kmeans
dataset.head()
```

OR
```python

# Credit Card Cluster Problem with K-Means
# Importing Preprocessing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Importing Datasets
dataset = pd.read_csv('../input/CC GENERAL.csv')
X = dataset.iloc[:, 1:].values
# Dataset Contains Multiple Missing values
# Replacing Missing Value by Most Repeated/Frequent Number in that column
# Use Imputer with strategy 'most_frequent'
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="most_frequent", axis=0)
imputer = imputer.fit(X)
X = imputer.transform(X)
# Applying Feature Scalling with StandardScaler
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
# For Finding Optimal Number of Cluster use Elbow Method 
from sklearn.cluster import KMeans
wcss = []
for i in range(1,18):
 kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state
=0).fit(X)
 wcss.append(kmeans.inertia_)
 
plt.plot(range(1,18), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Cluster')
plt.ylabel('WCSS')
plt.show()
# Apply K-Means Again With Optimal Number of Cluster that we got from Elbow method i.e. 8
kmeans = KMeans(n_clusters=8, init='k-means++', max_iter=300, n_init=10, random_state=
0)
y_kmeans = kmeans.fit_predict(X)
# Finally Append new Column i.e Cluster to Actual Dataset
dataset['Cluster'] = y_kmeans
dataset.head()

```

## Slip 5

Q.Write a python program to implement hierarchical clustering algorithm.

```python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv('./Mall_customers.csv')
X = dataset.iloc[:,[3,4]].values
y = dataset.iloc[:, 3].values
X

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
# Assuming y_train is a 1D array
y_train = np.array(y_train).reshape(-1, 1)
y_train = sc_y.fit_transform(y_train)

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)


# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
```

## Slip 6

Write a python program to implement complete data pre-processing in a given data set (missmg value, encoding categorical value, Splitting the dataset into the training and test sets and feature scaling.(Download dataset from gtthub.com).

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
from sklearn.impute import SimpleImputer

# Load the diabetes dataset (already available in scikit-learn)
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Introduce missing values in the dataset (replace 10% of values with NaN)
import numpy as np
np.random.seed(42)
missing_mask = np.random.rand(X.shape[0], X.shape[1]) < 0.1
X[missing_mask] = np.nan

# Display the dataset with missing values
df = pd.DataFrame(data=np.c_[X, y], columns=[f'feature_{i}' for i in range(X.shape[1])] + ['target'])
print("Dataset with Missing Values:")
print(df.head())

# Handling missing values using SimpleImputer (replace NaN with mean)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Display the dataset after handling missing values
df_imputed = pd.DataFrame(data=np.c_[X_imputed, y], columns=[f'feature_{i}' for i in range(X.shape[1])] + ['target'])
print("\nDataset after Handling Missing Values:")
print(df_imputed.head())

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Display the shapes of training and test sets
print("\nShapes of Training and Test Sets:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display the scaled features
print("\nScaled Features:")
print("X_train_scaled:")
print(X_train_scaled[:5, :])  # Displaying the first 5 rows

print("\nX_test_scaled:")
print(X_test_scaled[:5, :])  # Displaying the first 5 rows
```


## Slip 7

Write a Python program to "StudentsPerformance.csv" file.
Solve following:
- To display the shape of dataset.
- To display the top rows of the dataset with their columns.
- To display the number of rows randomly
- To display the number of columns and names of the columns.
Note: Download dataset from following link
(https://wvvw.kaggle.cowspscientist'students-performance-in-exams?select=StudentsPerformance.csv)

```python
import pandas as pd

# Load the dataset from the provided link
url = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
df = pd.read_csv('./StudentsPerformance.csv')

# Display the shape of the dataset
print("Shape of the dataset:")
print(df.shape)

# Display the top rows of the dataset with their columns
print("\nTop rows of the dataset:")
print(df.head())

# Display a random sample of rows
num_random_rows = 5
print(f"\nRandom {num_random_rows} rows of the dataset:")
print(df.sample(num_random_rows))

# Display the number of columns and names of the columns
num_columns = df.shape[1]
column_names = df.columns.tolist()
print(f"\nNumber of columns: {num_columns}")
print("Column names:")
print(column_names)
```