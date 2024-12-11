#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction

# # Importing Required Librarires 

# In[ ]:


import pandas as pd   #used for accessing dataset
import numpy as np
import matplotlib.pyplot as plt  #ploting graphs
import seaborn as sns #used for ploting advance graphs
from sklearn.model_selection import train_test_split # for trainset
from sklearn.linear_model import LinearRegression 
from sklearn import metrics#This will used at the end for calculating accuracy.


# # Loading The dataset

# In[6]:


data = pd.read_csv(r"E:\DS_Assignment\DS_ML_Pro\USA_Housing.csv")
data


# In[32]:


data.info()


# In[33]:


data.describe()


# # Drop Address Column

# In[7]:


data =data.drop(['Address'], axis = 1)
data.head()


# # Check For Missing Data

# In[34]:


print(data.isnull().sum())


# In[8]:


sns.heatmap(data.isnull())


# In[35]:


sns.pairplot(data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 
                   'Avg. Area Number of Bedrooms', 'Area Population', 'Price']])
plt.show()


# # Training and Predicting

# In[36]:


#features:(dependent Variable)
X = data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 
          'Avg. Area Number of Bedrooms', 'Area Population']]
#Target : (independent Variable)
y = data['Price']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[24]:


print(data.columns)


# In[37]:


# Initialize the model
model = LinearRegression()

# Train the model
model.fit(x_train, y_train)


# # Model Evaluation 

# In[38]:


# Predict on the test set
predictions = model.predict(x_test)


# In[39]:


# Calculate Mean Absolute Error
mae = metrics.mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error (MAE): {mae}")

# Calculate Mean Squared Error
mse = metrics.mean_squared_error(y_test, predictions)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate Root Mean Squared Error
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")


# # Visualize the Results

# In[40]:


# Scatter plot to compare actual vs predicted prices
plt.scatter(y_test, predictions, color="blue")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()


# In[41]:


# Plot distribution of residuals (errors)
sns.histplot(y_test - predictions, kde=True)
plt.title("Distribution of Residuals")
plt.xlabel("Residuals")
plt.show()


# # Evaluation 

# In[42]:


coefficients = model.coef_
intercept = model.intercept_
print("Coefficients:", coefficients)
print("Intercept:", intercept)


# In[44]:


predictions = model.predict(x_test)
print("Predicted Prices for Test Data:", predictions[:5])


# In[46]:


# Calculate an Example Prediction Manually Using Coefficients and Intercept
# Select the first row from X_test to calculate manually
sample_data = x_test.iloc[0]
manual_prediction = (
    coefficients[0] * sample_data['Avg. Area Income'] +
    coefficients[1] * sample_data['Avg. Area House Age'] +
    coefficients[2] * sample_data['Avg. Area Number of Rooms'] +
    coefficients[3] * sample_data['Avg. Area Number of Bedrooms'] +
    coefficients[4] * sample_data['Area Population'] +
    intercept
)
print("Manual Prediction for First Test Data Row:", manual_prediction)
print("Model Prediction for First Test Data Row:", predictions[0])


# In[ ]:




