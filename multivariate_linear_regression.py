# %% [markdown]
# # Importing Libraries #

# %%
# Import libraries necessary
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# %% [markdown]
# # Load the data #

# %%
# Read the csv file
df_data = pd.read_csv("realestate.csv")

# Remove the 'No' column
df_data.drop('No', inplace=True, axis=1)

# %%
# Print info about the dataset
df_data.head()
df_data.info()

# Declare the dependent variables
features = df_data.drop('Y house price of unit area', axis=1).columns
print("features : ", features)

# %% [markdown]
# # Plotting the independent and dependent variables #

# %%
sns.pairplot(df_data, x_vars=features, y_vars='Y house price of unit area')

# %% [markdown]
# # Plotting the heatmap #

# %%
sns.heatmap(df_data.corr(), annot=True)

# %% [markdown]
# # Using Linear Regression #

# %%
from sklearn.linear_model import LinearRegression

LinearModel = LinearRegression()
x = df_data.drop('Y house price of unit area', axis=1)
y = df_data['Y house price of unit area']
LinearModel.fit(x, y)

print(LinearModel.intercept_)
print(LinearModel.coef_)

list(zip(features, LinearModel.coef_))

# %% [markdown]
# ## Calculating the linear regression model #1 accuracy ##

# %%
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

LinearModel_Predictions = LinearModel.predict(x)

print("mean_squared_error : ", mean_squared_error(y, LinearModel_Predictions))  
print("R2 : ", r2_score(y, LinearModel_Predictions))

# %% [markdown]
# ## Plotting the actual vs predicted values on a graph ##

# %%
plt.scatter(y, LinearModel_Predictions)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values for House Price")
plt.show()


