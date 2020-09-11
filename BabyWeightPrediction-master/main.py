# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Prediction of Baby Weight
# 
# * Author: Shreya Dubey 
# %% [markdown]
# ### TASK 1: Import all the necessary packages here
# 

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
import sklearn
from sklearn.model_selection import train_test_split
import warnings
import math
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=DeprecationWarning)

# %% [markdown]
# ### TASK 2: Load the dataset into memory so that you can play with it here

# %%
df_train = pd.read_csv("baby-weights-dataset2.csv")

# %% [markdown]
# ### TASK 3: Compute mean, stdev, min, max, 25% percentile, median and 75% percentile of the dataset (BWEIGHT variable)

# %%
print (df_train.shape)
df_train.head
df_train['BWEIGHT'].describe()

# %% [markdown]
# ### TASK 4: Also, draw the histogram plot for the BWEIGHT variable
# 

# %%
sns.distplot(df_train['BWEIGHT'])

# %% [markdown]
# ### TASK 5: Present the skewness and kurtosis of the BWEIGHT target variable

# %%
df_train['BWEIGHT_Log']= np.log(df_train['BWEIGHT'])

sns.distplot(df_train['BWEIGHT_Log'])

print("Skewness: %f" % df_train['BWEIGHT_Log'].skew())
print("Kurtosis: %f" % df_train['BWEIGHT_Log'].kurt())

# %% [markdown]
# ### TASK 6: Do variable selection from the pool of 124 variables based on correlation score with the target variable BWEIGHT 
# %% [markdown]
# I am choosing 11 variables as my training variables as they have a correlation score above 0.09.
# These variables are - WEEKS, GAINED, VISITS, HYPERPR, MARITAL, SEX, CIGNUM, RACEDAD, RACEMOM, PRETERM and MAGE

# %%
corr_matrix = df_train.corr()
corr_abs = corr_matrix.abs()

#set the figure size for heatmap 
plt.figure(figsize = (12,10))

#plot heatmap 
sns.heatmap(corr_matrix, xticklabels = corr_matrix.columns.values, yticklabels=corr_matrix.columns.values)

print(corr_abs['BWEIGHT'].sort_values(ascending=False))

#from pandas.tools.plotting import scatter_matrix
#attributes = ["BWEIGHT", "WEEKS", "GAINED", "VISITS", "HYPERPR","MARITAL", "SEX", "CIGNUM", "RACEDAD", "RACEMOM"]
#scatter_matrix(df_train[attributes], figsize=(16,16))

# %% [markdown]
# ### TASK 7: Check for missing data, and tackle it
# * There is no missing data as from the info we see that all columns have eaqual number of elements
# * Still I have used the imputer function to replace any empty value by meadian value.

# %%
df_train.info()
df_train_num1 = df_train.drop("HISPMOM", axis=1)
df_train_num2 = df_train_num1.drop("HISPDAD", axis=1)

from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = "median")
imputer.fit(df_train_num2)

df_train_num2.head

# %% [markdown]
# ### TASK 8: Tackle the dummy categorical variables by introducing dummy variables

# %%
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df_train_dum = df_train["HISPMOM"]
df_train_dum_encoded = encoder.fit_transform(df_train_dum)

df_train_dum1 = df_train["HISPDAD"]
df_train_dum1_encoded = encoder.fit_transform(df_train_dum1)

df_train['HISPMOM'] = df_train_dum_encoded
df_train['HISPDAD'] = df_train_dum1_encoded
df_train.shape

# %% [markdown]
# ### TASK 9.1: Randomly split the dataset into training, Tr (80%) and testing, Te (20%)

# %%
train_set, test_set = train_test_split(df_train, test_size=0.20)

print(train_set.shape)
print(test_set.shape)

corr_matrix = df_train.corr()
corr_abs = corr_matrix.abs()
print(corr_abs['BWEIGHT'].sort_values(ascending=False))

# %% [markdown]
# ### TASK 9.2: On the training dataset, apply a normalization technique

# %%
def minMax(train_set):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(train_set)
    scaler.transform(train_set)
    return train_set

def standarize(train_set):
    from sklearn.preprocessing import StandardScaler
    standardscaler = StandardScaler()
    standardscaler.fit(train_set)
    standardscaler.transform(train_set)
    return train_set


# %%
#train_set = standarize(train_set)

train_set = minMax(train_set)


xtrain_set = train_set[["WEEKS", "GAINED", "VISITS", "HYPERPR", "MARITAL", "SEX", "CIGNUM", "RACEDAD", "RACEMOM", "PRETERM", "MAGE"]]
ytrain_set = train_set[["BWEIGHT"]]

# %% [markdown]
# ### TASK 9.3: Apply the training data statistics to normalize the testing data as well.

# %%


test_set = minMax(test_set)

xtest_set = test_set[["WEEKS", "GAINED", "VISITS", "HYPERPR", "MARITAL", "SEX", "CIGNUM", "RACEDAD", "RACEMOM", "PRETERM", "MAGE"]]
ytest_set = test_set[["BWEIGHT"]]

# %% [markdown]
# ### TASK 10: Find the linear regression function describing the training dataset using Gradient Descent (batch or stochastic or mini-batch).

# %%
#converts pandas Dataframe to numpy.ndarray
X = xtrain_set.iloc[:,:]
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)
y = ytrain_set.iloc[:,:].values 

#select initial weights. Taking all zeros to start with
weight = np.zeros([1,X.shape[1]])

#set learning rate and number of iterations
alpha = 0.00055
iters = 20000

# Print shapes of X, y and theta to make sure their dimensions are correct 
print(X.shape) ; 
print(y.shape) ;
print(weight.shape);

#computeLost
def computeLoss(X,y,weight):
    
    h = X@weight.T    # '@' for matrix multiplication. Alternatively matmult() can be used
    error = h-y    
    loss = np.power(error,2)
    J = np.sum(loss)/(2*len(X))
    return J

print(computeLoss(X,y,weight))

#gradient descent
def gradientDescent(X,y,weight,iters,alpha):
    loss = np.zeros(iters)
    for i in range(iters):
        y_hat = X@weight.T
        gradient = np.sum(X*(y_hat - y) ,  axis = 0)       
        weight = weight - (alpha/len(X))*gradient 
        loss[i] = computeLoss(X, y, weight) 
        
        # print cost after every 100 iterations to keep a track of when the cost function converges
        if i%100 == 0:
            print("Iteration %d | loss: %f" % (i, loss[i]))
            
    return weight,loss

#running the gd and cost function
W,loss = gradientDescent(X,y,weight,iters,alpha)
print("Weigths for the linear model", W)

finalCost = computeLoss(X,y,W)
print(finalCost)

# %% [markdown]
# ### Task 11: Predict BWEIGHT target variable for each of the testing dataset using the regression line learned in Task 10, and report RMSE(testing) (Root Mean Squared Error)

# %%
# get all rows and columns of pandas dataframe xtest_set into numpy array X_test
X_test = xtest_set.iloc[:,:]

# make the first column all ones
ones = np.ones([X_test.shape[0],1])
X_test = np.concatenate((ones,X_test),axis=1)

#get all rows and columns of ytest_set into numpy array y_test
y_test = ytest_set.iloc[:,:].values 

print("shape of X_test: ", X_test.shape)
print("shape of W: ", W.shape)
print("shape of y_test: ", y_test.shape)


#err = y_test - np.matmul(X_test,W.T)  
def getRMSE(y_test,x_test,W):
    err = y_test - (x_test@W.T)  
    sq_err = err*err ; 
    mean_sqr_err = np.sum(sq_err, axis = 0 )/len(err)
    RMSE = math.sqrt(mean_sqr_err)
    return RMSE

rmse1 = getRMSE(y_test,X_test,W)
print("Root Mean Squared Error = ", rmse1)

# %% [markdown]
# ### Repeat TASK 10 additional four times : Run linear regression training again
# ### After each run, Report RMSE(testing)
# * RMSE 1 = 1.089435890861352
# * RMSE 2 = 1.0769502778804536
# * RMSE 3 = 1.0751931372890815
# * RMSE 4 = 1.0838836320175358
# * RMSE 5 = 1.076509556436053
# * RMSE 6 = 1.079073176421531
# * RMSE 7 = 1.0830120696681553
# * RMSE 8 = 1.0769365485793123
# * RMSE 9 = 1.0819189516943706
# * RMSE 10 = 1.083887357731135
# * RMSE 11 = 1.0691456953936629
# * RMSE 12 = 1.0798605679268696
# * RMSE 13 = 1.0694789612837972
# * RMSE 14 = 1.0821356386698318
# * RMSE 15 = 1.086356484150816

# %%
rmseValues = np.array([1.089435890861352,1.0769502778804536,1.0751931372890815,1.0838836320175358,1.076509556436053])
rmseValues2 = np.array([1.079073176421531,1.0830120696681553,1.0769365485793123,1.0819189516943706,1.083887357731135])
rmseValues3 = np.array([1.0691456953936629,1.0798605679268696,1.0694789612837972,1.0821356386698318,1.086356484150816])

# %% [markdown]
# ### Task 12: Finally, Report RMSE(testing) = Average(RMSE_test) $\pm$ Stdev(RMSE_test)
# ### Here Average(RMSE_test) = average of all the 5 RMSE(testing) scores you got above.
# ### And, stdev(RMSE_test) = standard deviation of all the 5 RMSE(testing) scores above.

# %%
avgRMSE = np.mean(rmseValues)
stdevRMSE = np.std(rmseValues)

avgRMSE2 = np.mean(rmseValues2)
avgRMSE3 = np.mean(rmseValues3)

RMSE_avg_set = np.array([rmseValues,rmseValues2,rmseValues3])

stdevRMSE2 = np.std(rmseValues2)
stdevRMSE3 = np.std(rmseValues3)

RMSE_stdev_set = np.array([stdevRMSE,stdevRMSE2,stdevRMSE3])

RMSE_testing1 = avgRMSE + stdevRMSE
RMSE_testing2 = avgRMSE -stdevRMSE

print("Average RMSE = ",avgRMSE)

print("RMSE_testing1 = ",RMSE_testing1)
print("RMSE_testing2 = ",RMSE_testing2)

# %% [markdown]
# ### Task 13: Run linear regression one last time on the whole dataset  (i.e, training+testing which is preprocessed by you above).

# %%
#df_train = minMax(df_train)
df_train = standarize(df_train)

x_full_set = df_train[["WEEKS", "GAINED", "VISITS", "HYPERPR", "MARITAL", "SEX", "CIGNUM", "RACEDAD", "RACEMOM", "PRETERM", "MAGE"]]
y_full_set = df_train[["BWEIGHT"]]

print(x_full_set.shape)

X_full = x_full_set.iloc[:,:]
ones = np.ones([X_full.shape[0],1])
X_full = np.concatenate((ones,X_full),axis=1)

y_full = y_full_set.iloc[:,:].values #.converts pandas Dataframe to numpy.ndarray

#select initial weights. Taking all zeros to start with
weight = np.zeros([1,X_full.shape[1]])

W_new,loss_new = gradientDescent(X_full,y_full,weight,iters,alpha)
print("Weigths for the linear model", W)

# %% [markdown]
# ### Task 14: Preprocess the judge-without-label.csv file according to the strategy applied above on the whole dataset (task 13)

# %%
#Reading the judge-without-label data
df_judge = pd.read_csv("judge-without-labels_2.csv")

xjudge_set = df_judge[["WEEKS", "GAINED", "VISITS", "HYPERPR", "MARITAL", "SEX", "CIGNUM", "RACEDAD", "RACEMOM", "PRETERM", "MAGE"]]

# get all rows and columns of pandas dataframe xtest_set into numpy array X_test
X_Judge = xjudge_set.iloc[:,:]

# make the first column all ones
ones = np.ones([X_Judge.shape[0],1])
X_Judge = np.concatenate((ones,X_Judge),axis=1)

X_Judge = standarize(X_Judge)
#X_Judge = minMax(X_Judge)

# %% [markdown]
# ### Task 15: Predict BWEIGHT for each of the samples from the judge-without-label.csv file

# %%
print("shape of X_test: ", X_Judge.shape)
print("shape of W: ", W.shape)

BWEIGHT_PREDICION =  (X_Judge@W.T)  

print(BWEIGHT_PREDICION.shape)

dataset = pd.DataFrame({'BWEIGHT':BWEIGHT_PREDICION[:,0]})

dataset.to_csv('BWEIGHT_PREDICTION.csv', sep=',')

print(BWEIGHT_PREDICION.shape)

# %% [markdown]
# ### Task 16: Repeat tasks 9-12 three times, and report the ultimate RMSE_test average $\pm$ ultimate RMSE_test stdev

# %%
ultimate_test_avg = np.mean(RMSE_avg_set)
ultimate_test_stdev = np.mean(RMSE_stdev_set)

print("RMSE_test average + ultimate RMSE_test stdev = ",ultimate_test_avg+ultimate_test_stdev)
print("RMSE_test average - ultimate RMSE_test stdev = ",ultimate_test_avg-ultimate_test_stdev)
print("The Weight = ",((ultimate_test_avg+ultimate_test_stdev)+(ultimate_test_avg-ultimate_test_stdev)/2))

# %% [markdown]
# 
# 
# #Run1 RMSE = 1.07638
# #Run2 RMSE = 1.06754
# #Run3 RMSE = 1.05632
# #Run4 RMSE = 1.05414
# #Run5 RMSE = 1.05414
# #Run6 RMSE = 1.06651
# #Run7 RMSE = 1.06642
# #Run8 RMSE = 1.06233
# 
# #Kaggle Handle ---> Hariharan 
# 
# 

# %%



