# %% [markdown]
# # Importing Libraries #

# %%
# Import libraries necessary
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
# %matplotlib inline

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

# %% [markdown]
# # Plotting the independent and dependent variables #

# %%
sns.pairplot(df_data, x_vars=features, y_vars='Y house price of unit area', aspect=1, height=5)

# %% [markdown]
# # Plotting the heatmap #

# %%
sns.heatmap(df_data.corr(), annot=True)

# %% [markdown]
# # Using Linear Regression model #1 #

# %%
from sklearn.linear_model import LinearRegression

x = df_data.drop('Y house price of unit area', axis=1)
y = df_data['Y house price of unit area']

LinearModel = LinearRegression()
LinearModel.fit(x, y)

# Print the intercept for the model
# This is sort of a 'base guess' for the price of a house
print(LinearModel.intercept_)

# Print the coefficients for each dependent variable
# You could say that for each 1 unit increase in a variable, house price increases or decreases according to the coefficient value 
print(LinearModel.coef_)

# Print the coefficients witht heir labels
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

# %% [markdown]
# # Using Linear Regression model #2 #

# %%
from sklearn.linear_model import SGDRegressor

SGDModel = SGDRegressor(learning_rate="constant", max_iter=10000, tol=1e-3)
x = df_data.drop('Y house price of unit area', axis=1)
y = df_data['Y house price of unit area']

# %% [markdown]
# ### Scaling the data ###

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# %% [markdown]
# ### Computing for the intercept and coefficients ###

# %%
SGDModel.fit(x_scaled, y)

# Print the intercept for the model
# This is sort of a 'base guess' for the price of a house
print("SGD Regressor Intercept : ", SGDModel.intercept_)

# Print the coefficients for each dependent variable
# You could say that for each 1 unit increase in a variable, house price increases or decreases according to the coefficient value 
print("SGD Regressor Coefficient : ", SGDModel.coef_)

# Print the coefficients witht heir labels
list(zip(features, SGDModel.coef_))

# %% [markdown]
# ### Calculating the linear model #2 accuracy ###

# %%
SGD_Predictions = SGDModel.predict(x_scaled)

print("mean_squared_error : ", mean_squared_error(y, SGD_Predictions))  
print("R2 : ", r2_score(y, SGD_Predictions))

# %% [markdown]
# ## Plotting the graph for the model #2 ##

# %%
plt.scatter(y, SGD_Predictions)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values for House Price")
plt.show()


