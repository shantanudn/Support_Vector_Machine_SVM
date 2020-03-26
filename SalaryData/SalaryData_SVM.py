import pandas as pd 
import numpy as np 
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

salary_train = pd.read_csv("C:/Training/Analytics/Support_Vector_Machine/SalaryData/SalaryData_Train.csv")
salary_test = pd.read_csv("C:/Training/Analytics/Support_Vector_Machine/SalaryData/SalaryData_Test.csv")
salary_train.head()
describe = salary_train.describe()
salary_train.columns


# =============================================================================
# Preprocessing categorical data
# =============================================================================

string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]
from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in string_columns:
    salary_train[i] = number.fit_transform(salary_train[i])
    salary_test[i] = number.fit_transform(salary_test[i])

describe = salary_train.describe()



# =============================================================================
# Normalization:
# =============================================================================
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(salary_train.iloc[:,0:13])
nor_des = df_norm.describe()

df_norm_test = norm_func(salary_test.iloc[:,0:13])

salary = salary_train['Salary']
salary_tcol = salary_test['Salary']

salary_train = pd.concat([df_norm,salary],axis=1)

salary_test = pd.concat([df_norm_test,salary_tcol],axis=1)
# =============================================================================
# =============================================================================
# # EDA
# =============================================================================
# =============================================================================

sns.boxplot(x="age",y="workclass",data=salary_train,palette = "hls")
sns.boxplot(x="y-box",y="lettr",data=salary_train,palette = "hls")

# =============================================================================
# Histogram
# =============================================================================
salary_train.hist()


# =============================================================================
# Continous distribution funtion
# =============================================================================
salary_train.plot(kind='density', subplots=True, layout=(4,4), sharex=False, sharey=False)


# =============================================================================
# boxplot
# =============================================================================
salary_train.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)


# =============================================================================
# ScatterPlot
# =============================================================================
scatter_matrix(salary_train)

# =============================================================================
# Corrlation Heat Map
# =============================================================================

heat1 = salary_train.corr()
sns.heatmap(heat1, xticklabels=salary_train.columns, yticklabels=salary_train.columns, annot=True)


# =============================================================================
# Heat map without values
# =============================================================================


sns.pairplot(data=salary_train)

from sklearn.svm import SVC
#from sklearn.model_selection import train_test_split
#train,test = train_test_split(salary_train,test_size = 0.3)
salary_test.head()
salary_train_X = salary_train.iloc[:,0:13]
salary_train_Y = salary_train.iloc[:,13]
salary_test_X  = salary_test.iloc[:,0:13]
salary_test_y  = salary_test.iloc[:,13]

# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(salary_train_X,salary_train_Y)
pred_test_linear = model_linear.predict(salary_test_X)

np.mean(pred_test_linear==salary_test_y) # Accuracy = 80.92

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(salary_train_X,salary_train_Y)
pred_test_poly = model_poly.predict(salary_test_X)

np.mean(pred_test_poly==salary_test_y) # Accuracy = 84.06

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(salary_train_X,salary_train_Y)
pred_test_rbf = model_rbf.predict(salary_test_X)

np.mean(pred_test_rbf==salary_test_y) # Accuracy = 84.65

# kernel = sigmoid
model_sigmoid = SVC(kernel = "sigmoid")
model_sigmoid.fit(salary_train_X,salary_train_Y)
pred_test_sigmoid = model_sigmoid.predict(salary_test_X)

np.mean(pred_test_sigmoid==salary_test_y) # Accuracy = 75.28

