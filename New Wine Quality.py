import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.model_selection import train_test_split


# plt.rc("font", size=14)
import seaborn as sns
# sns.set(style="white") #/white background style for seaborn plots
# sns.set(style="whitegrid", color_codes=True)
# from ggplot import *

#Load wine quality data

df_red = pd.read_csv('winequality_red.csv')
df_white = pd.read_csv('winequality_white.csv')
#to make all columns show in
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)


df_red['color'] = 'R'
df_white['color'] = 'W'

df_all = pd.concat([df_red,df_white],axis=0)
# print(df_all.head(10))

df_white.rename(columns={'fixed acidity': 'fixed_acidity','citric acid':'citric_acid',
                         'volatile acidity':'volatile_acidity','residual sugar':'residual_sugar'
                         ,'free sulfur dioxide':'free_sulfur_dioxide',
                         'total sulfur dioxide':'total_sulfur_dioxide'},inplace=True)
df_red.rename(columns={'fixed acidity': 'fixed_acidity','citric acid':'citric_acid','volatile acidity':'volatile_acidity',
                       'residual sugar':'residual_sugar','free sulfur dioxide':'free_sulfur_dioxide',
                       'total sulfur dioxide':'total_sulfur_dioxide'}, inplace=True)

df_all.rename(columns={'fixed acidity': 'fixed_acidity','citric acid':'citric_acid','volatile acidity':'volatile_acidity',
                       'residual sugar':'residual_sugar','free sulfur dioxide':'free_sulfur_dioxide',
                       'total sulfur dioxide':'total_sulfur_dioxide'}, inplace=True)

# print(df_all.head())
# print(df_all.isnull().sum())
# print(df_all.describe())
# df = pd.get_dummies(df_all,columns=['colour'])

# print('white mean = ',df_white['quality'].mean())
# print('red mean = ',df_red['quality'].mean())

d = {'color':['red','white'],'mean_quality':[5.636023,5.877909]}
df_mean = pd.DataFrame(data=d)
# print(df_mean)
plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("Wine Characteristic Correlation Heatmap (Reds)")
corr = df_red.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
           cmap="Reds")
# plt.show()

plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("Wine Characteristic Correlation Heatmap (Reds)")
corr = df_red.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
           cmap="Purples")
# plt.show()

#biggest difference between white and red correlations

df_r_corr=df_red.corr()
df_w_corr=df_white.corr()
# print(df_r_corr)
# print(df_w_corr)
diff_corr = df_w_corr - df_r_corr
plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title('Correlation difference between red and white wines')
corr = diff_corr
sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values,cmap='coolwarm')
# plt.show()

