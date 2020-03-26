import pandas as pd 
import numpy as np 
import seaborn as sns

forestfires = pd.read_csv("C:/Training/Analytics/Support_Vector_Machine/forestfires/forestfires.csv")

########################## Neural Network for predicting continuous values ###############################

# Reading data 
forestfires = pd.read_csv("C:/Training/Analytics/Support_Vector_Machine/forestfires/forestfires.csv")
forestfires.head()

forestfires_ori = forestfires

forestfires = forestfires[['month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area','size_category']].copy()
forestfires.columns


# Encode Data
forestfires.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
forestfires.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)

##### EDA ###################



print("Head:", forestfires.head())



describe =  forestfires.describe()

forestfires.columns

forestfires.dtypes

# =============================================================================
# Normalization:
# =============================================================================
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(forestfires.iloc[:,0:11])
nor_des = df_norm.describe()

forest_size = forestfires['size_category']

forestfires = pd.concat([df_norm,forest_size],axis=1)


print("Shape:", forestfires.shape)

print("Data Types:", forestfires.dtypes)

print("Correlation:", forestfires.corr(method='pearson'))



dataset = forestfires.values


X = dataset[:,0:11]
Y = dataset[:,11]



# =============================================================================
# Histogram
# =============================================================================
forestfires.hist()


# =============================================================================
# Continous distribution funtion
# =============================================================================
forestfires.plot(kind='density', subplots=True, layout=(4,4), sharex=False, sharey=False)



forestfires.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)

# =============================================================================
# Pairplot or scattermatrix
# =============================================================================

sns.pairplot(data=forestfires)

# =============================================================================
# Corrlation Heat Map
# =============================================================================

heat1 = forestfires.corr()
sns.heatmap(heat1, xticklabels=forestfires.columns, yticklabels=forestfires.columns, annot=True)



# =========================

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test = train_test_split(forestfires,test_size = 0.3)
test.head()
train_X = train.iloc[:,0:11]
train_y = train.iloc[:,11]
test_X  = test.iloc[:,0:11]
test_y  = test.iloc[:,11]

# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

# kernel = linear
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear==test_y) # Accuracy = 91%

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)

np.mean(pred_test_poly==test_y) # Accuracy = 78%

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y) # Accuracy = 78.84%

# kernel = sigmoid
model_sigmoid = SVC(kernel = "sigmoid")
model_sigmoid.fit(train_X,train_y)
pred_test_sigmoid = model_sigmoid.predict(test_X)

np.mean(pred_test_sigmoid==test_y) # Accuracy = 76.92


