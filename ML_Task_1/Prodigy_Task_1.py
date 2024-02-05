# This is the code for Task 1 Prodigy Techinfo internship
# Task 1: House Price Prediction
# The Code is written by Muhammad Mudassir Majeed
# The date is Jan-24.
# Dataset: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

#-----------------------------------------------------------------------------#

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Part 1: Load Dataset
data_train = pd.read_csv(r'Task 1\train.csv')
data_test = pd.read_csv(r'Task 1\test.csv')

# Part 2: Preliminary EDA
data_train.info()
data_train.head()
data_test.info()
data_test.head()

# Check for Null Values
data_train.isnull().sum() # Some columns have Null values
data_test.isnull().sum()  # Some Columns have Null values

# Check for Duplicate values
data_train.duplicated().sum() # No Duplicates
data_test.duplicated().sum()  # No Duplicates 

# Part 3: Pre-processing
# We want prediction based only on house size, number of washrooms and number of bathrooms
# We will combine relevant columns and drop unnecessary ones.

# Drop unneeded columns
drop_col_1 = ['Id','MSSubClass','MSZoning','LotFrontage','Street','Alley','LotShape',
            'LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
            'Condition1','Condition2','BldgType','HouseStyle','OverallQual',
            'OverallCond','YearBuilt','YearRemodAdd','RoofStyle','RoofMatl',
            'Exterior1st','Exterior2nd']

drop_col_2 = ['MasVnrType','MasVnrArea','ExterQual','ExterCond','Foundation',
              'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1',
              'BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','Heating',
              'HeatingQC','CentralAir','Electrical','1stFlrSF','2ndFlrSF','LowQualFinSF',
              'GrLivArea','KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional',
              ]

drop_col_3 = ['Fireplaces','FireplaceQu','GarageType','GarageYrBlt','GarageFinish',
              'GarageCars','GarageArea','GarageQual','GarageCond','PavedDrive','WoodDeckSF',
              'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','PoolQC',
              'Fence','MiscFeature','MiscVal','MoSold','YrSold','SaleType','SaleCondition']

data_train = data_train.drop(drop_col_1, axis =1)
data_train = data_train.drop(drop_col_2, axis =1)
data_train = data_train.drop(drop_col_3, axis =1)
data_train.info()

data_test = data_test.drop(drop_col_1, axis =1)
data_test = data_test.drop(drop_col_2, axis =1)
data_test = data_test.drop(drop_col_3, axis =1)
data_test.info()

# Combine Bath room columns to make a single total bathrooms column
data_train['TotalBath'] = data_train[['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath']].sum(axis=1)
data_test['TotalBath'] = data_test[['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath']].sum(axis=1)

data_train = data_train.drop(['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath'], axis = 1)
data_test = data_test.drop(['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath'], axis = 1)

data_train.info()
data_test.info()

# Box Plots for outliers
grid_row, grid_col = 2,2
plt.figure(figsize=(12,10))
plt.suptitle('Box Plots')

for i, column in enumerate(data_train.columns, 1):
    plt.subplot(grid_row,grid_col,i)
    sns.boxplot(data=data_train[column])
    plt.title(f'Plot for {column}')
    
plt.tight_layout(rect=([0.00,0.03,0.99,1]))
plt.show()

grid_row, grid_col = 2,2
plt.figure(figsize=(12,10))
plt.suptitle('Box Plots')

for i, column in enumerate(data_test.columns, 1):
    plt.subplot(grid_row,grid_col,i)
    sns.boxplot(data=data_test[column])
    plt.title(f'Plot for {column}')
    
plt.tight_layout(rect=([0.00,0.03,0.99,1]))
plt.show()

# We have large values for outliers.
# We will leave them as is

# Split into X and Y
X = data_train.drop(['SalePrice'], axis =1)
Y = data_train['SalePrice']

# Standardize Data
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_stand = scalar.fit_transform(X)
X_std = pd.DataFrame(data = X_stand,columns = X.columns)

# Part 4: Model Training
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_std,Y, test_size=0.2, random_state=42)
 
# Random Forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train,Y_train)

Y_predict_rf = rf.predict(X_test)

# Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)

Y_predict_lr = lr.predict(X_test)

# KNN
from sklearn.neighbors import KNeighborsRegressor
KNN = KNeighborsRegressor(n_neighbors= 4)
KNN.fit(X_train,Y_train)

Y_predict_KNN = KNN.predict(X_test)


# Part 5: Model Evaluation
from sklearn.metrics import r2_score, mean_squared_error

# r2_score
r2_rf = r2_score(Y_test,Y_predict_rf)
r2_lr = r2_score(Y_test,Y_predict_lr)
r2_KNN = r2_score(Y_test,Y_predict_KNN)

# MSE
MSE_rf = mean_squared_error(Y_test, Y_predict_rf)
MSE_lr = mean_squared_error(Y_test, Y_predict_lr)
MSE_KNN = mean_squared_error(Y_test, Y_predict_KNN)

# Data Presentation
from tabulate import tabulate

data = [['Random Forest', r2_rf, MSE_rf],
        ['Linear Regression', r2_lr, MSE_lr],
        ['KNN', r2_KNN, MSE_KNN]
        ]
header =['Model Name', 'r2_score','MSE_score']

table = tabulate(data, headers=header, tablefmt='fancy_grid',
                 floatfmt={'.0%','.2%','.2%'})
print(table)
