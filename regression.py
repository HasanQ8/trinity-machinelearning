# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 12:10:03 2019

@author: Hasan
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder,OneHotEncoder


#########################################################################

#feature selection


dataset = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')

#dropping missing values increased accuracy and got a lower mae
dataset = dataset.dropna()

#find outliers in the dataset for numerical and catagorical data
category_col =['Gender', 'Country','Profession', 'University Degree',
               'Hair Color'] 
for c in category_col:
    print (c)
    print (dataset[c].value_counts())
    
    
numeric_col =['Year of Record', 'Age', 'Body Height [cm]'] 
for c in numeric_col:
    print (c)
    print (dataset[c].value_counts())
    
#remove the outliers 
    
#dataset = dataset[~dataset['Country'].isin(['Argentina','Barbados', 'DR Congo'])]
dataset = dataset[~dataset['Profession'].isin(['Building Advisor','community supervisor','advertising and promotions manager'             
,'college aide'                                   
,'assistant civil engineer'                       
,'accessibility program manager'                  
,'animal trainer'                                 
,'accountant'                                     
,'coach'                                          
,'audit supervisor'                               
,'chief information security officer'             
,'biological technician'                          
,'contract manager'                               
,'building inspector'                             
,'cloud reliability engineer'                     
,'agricultural scientist'                         
,'Brewery Manager'                                
,'Bar Manager'                                    
,'compliance manager'                             
,'bridge operator'                                
,'assistant commissioner of enforcement'          
,'collector'                                      
,'community associate'                            
,'bill and account collector'                     
,'Blinds Installer'                               
,'computer aide'                                  
,'administrative procurement analyst'             
,'assistant commissioner of housing policy'       
,'administrative transportation coordinator'      
,'client services representative'])]

dataset = dataset[~dataset['Country'].isin(['Suriname'                  
,'Sudan'                      
,'Malta'
,'Algeria'                    
,'Micronesia'
,'Cabo Verde'                 
,'Brunei'                     
,'Argentina'                  
,'Ukraine'                    
,'Colombia'                   
,'Uganda'                     
,'Vanuatu'                    
,'Spain'                      
,'Barbados'                   
,'Kenya'                      
,'Belize'                     
,'DR Congo'                   
,'Saint Lucia'                
,'South Korea'                
,'Bahamas'                    
,'Tanzania'                   
,'Tonga'                      
,'Vietnam'                    
,'Myanmar'                    
,'Grenada'                    
,'United Kingdom'             
,'Seychelles'                 
,'Thailand'                   
,'South Africa'               
,'Kiribati'])]

           




#replace the catagorical gender cloumn with numerical data
    
    
#dataset = dataset[~dataset['Profession'].isin(['administrative contract specialist', 'administrative engineer', 'administrative services manager', 'administrator on duty' , 'agricultural technician', 'animal caretaker', 'appliance repairer', 'assessor','assistant commissioner of communications and policy', 'bartender', 'bartender helper', 'billing and posting clerk', 'biological scientist', 'biomedical engineer', 'call center manager', 'chauffeur' ])]
#dataset= dataset.dropna(subset=['Profession', 'Size of City', 'Wears Glasses', 'Body Height [cm]','Country','University Degree','Hair Color','Gender' ])
new_gender = {"Gender": {"male": 1, "female": 2, "other": 3, 'unknown':3, '0':3}}
dataset.replace(new_gender, inplace=True)

    
dataset.replace('0', np.nan, inplace= True)
dataset.replace('No', np.nan, inplace= True)
#dataset.replace('unknown', np.nan, inplace= True)


#dataset = dataset.dropna()

X = dataset.iloc[:,[1,2,3,4,6,10]].values
Y = dataset.iloc[:,11].values
X = pd.DataFrame(X)
X.head()






#transform missing data gotten from the outliers 0 and unknown
 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'constant')
imputer = imputer.fit(X.iloc[:,[3,4]])
X.iloc[:,[3,4]] = imputer.transform(X.iloc[:,[3,4]])
#imputer = imputer.fit(X.iloc[:,[1,3,5,6]])
#X.iloc[:,[1,3,5,6]] = imputer.transform(X.iloc[:,[1,3,5,6]])
 
imputer = SimpleImputer(missing_values = np.nan,strategy = 'median')
imputer= imputer.fit(X.iloc[:,[0,1,2,5]].values)
X.iloc[:,[0,1,2,5]] = imputer.transform(X.iloc[:,[0,1,2,5]])
#scaler = MinMaxScaler()
#X.iloc[:,[0,2]] = scaler.fit_transform(X.iloc[:,[0,2]])

#use labelendcoder and onehotencoder to tansform the catagorical data to arrays

labelencoder_X = LabelEncoder()
#X.iloc[:,1] = labelencoder_X.fit_transform(X.iloc[:,1])
X.iloc[:,3] = labelencoder_X.fit_transform(X.iloc[:,3])
#X.iloc[:,3] = labelencoder_X.fit_transform(X.iloc[:,3])
#X.iloc[:,5] = labelencoder_X.fit_transform(X.iloc[:,5])
X.iloc[:,4] = labelencoder_X.fit_transform(X.iloc[:,4])
#X.iloc[:,7] = labelencoder_X.fit_transform(X.iloc[:,7])



onehotencoder = OneHotEncoder(categorical_features =[3,4])
X = onehotencoder.fit_transform(X,Y).toarray()

labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)


#train and test the data 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.1, random_state=0)


#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

#use linear regression model to get prediction

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)

print('The training score for linear regression is:',(model.score(X_test, y_test)*100),'%')
y_pred=  model.predict(X_test)
mean_squared_error(y_pred,y_test)


#decision tree

model1 = DecisionTreeRegressor(max_leaf_nodes=500, random_state=0)
model1.fit(X_train, y_train)
print('The training score for DecisionTree regression is:',(model1.score(X_test, y_test)*100),'%')
#print('Validation accuracy', accuracy_score(y_test, regressor.predict(X_test)))
y_pred=  model1.predict(X_test)
mean_squared_error(y_pred,y_test)

#logistic Regression
model2 = LogisticRegression()
model2.fit(X_train,y_train)
#printing the training score
print('The training score for logistic regression is:',(model2.score(X_train,y_train)*100),'%')

#importing random forest classifier

model3 = RandomForestClassifier(n_estimators=6)
model3.fit(X_train,y_train)
#printing the training score
print('The training score for logistic regression is:',(model3.score(X_train,y_train)*100),'%')


#gradientboostingclassifier

model4 = GradientBoostingClassifier(n_estimators=5,learning_rate=1.1)
model4.fit(X_train,y_train)

print('The training score for logistic regression is:',(model4.score(X_train,y_train)*100),'%')

from sklearn.neighbors import KNeighborsClassifier
model5 = KNeighborsClassifier()
model5.fit(X_train, y_train)
model5.score(X_train, y_train)


#use ridge 

from sklearn.linear_model import Ridge

lr = LinearRegression()
lr.fit(X_train, y_train)
rr = Ridge(alpha=0.01)
rr.fit(X_train, y_train)
rr100 = Ridge(alpha=100) #comparison with alpha
rr100.fit(X_train, y_train)
train_score=lr.score(X_train, y_train)
test_score=lr.score(X_test, y_test)
Ridge_train_score = rr.score(X_train,y_train)
Ridge_test_score = rr.score(X_test, y_test)
Ridge_train_score100 = rr100.score(X_train,y_train)
Ridge_test_score100 = rr100.score(X_test, y_test)
print("linear regression train score:", train_score)
print("linear regression test score:", test_score)
print("ridge regression train score low alpha:", Ridge_train_score)
print("ridge regression test score low alpha:", Ridge_test_score)
print("ridge regression train score high alpha:", Ridge_train_score100)
print("ridge regression test score high alpha:", Ridge_test_score100)



#use regressor_ols to find the best features 
import statsmodels.formula.api as sm

X = np.append(arr = np.ones((90358, 1)).astype(float), values = X, axis = 1)
X_opt = X[:,:].values
X_opt = pd.DataFrame(X_opt)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
np.asarray(X_opt)
regressor_OLS.summary()
X_opt = X[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [2 ,3, 4, 5,6,7,8]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()


######################################################



#replace zero and unknown with empty 
dataset_new = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')

category_col =['Gender', 'Country','Profession', 'University Degree',
               'Hair Color'] 
for c in category_col:
    print (c)
    print (dataset_new[c].value_counts())

#dataset_new.replace('0', np.nan, inplace= True)
    
dataset_new.replace(['chief review officer'                                                 
,'application examiner'                                                  
,'audit supervisor'                                                      
,'clinical case supervisor'                                              
,'chief plan examiner'                                                   
,'administrative staff analyst'                                          
,'Branch Manager'                                                        
,'city research scientist'                                               
,'aerospace engineer'                                                    
,'bellhop'                                                               
,'commercial diver'                                                      
,'.net developer'                                                        
,'administrative associate to the executive director'                    
,'admin engineer'                                                        
,'atmospheric scientist'                                                 
,'cashier'                                                               
,'certified it administrator'                                            
,'assistant commissioner of the office of placement administration'      
,'chief operating officer'                                               
,'announcer'                                                             
,'building cleaner'                                                      
,'administrative aide'                                                   
,'back end developer'                                                    
,'animal trainer'                                                        
,'case management nurse'                                                 
,'clinical director'                                                     
,'automotive glass installer'                                            
,'confidential investigator'                                             
,'client services representative'                                        
,'accessibility outreach coordinator'], np.nan, inplace= True)
    
dataset_new.replace(['Montenegro'                 
,'Ukraine'                    
,'Sudan'                      
,'Uganda'                     
,'Iraq'                       
,'Cabo Verde'                 
,'Afghanistan'                
,'Colombia'                   
,'Iceland'                   
,'Vanuatu'                    
,'Brunei'                     
,'Algeria'                    
,'Malta'                      
,'Micronesia'                 
,'Spain'                      
,'South Korea'                
,'Myanmar'                    
,'Kenya'                      
,'Sao Tome & Principe'        
,'Maldives'                   
,'United Kingdom'             
,'Bahamas'                   
,'Belize'                    
,'France'                     
,'Italy'
,'Saint Lucia','Turkey'], np.nan, inplace= True)
#,'Saint Lucia','Turkey','Samoa','South Africa','Tanzania'
#dataset_new.iloc[:,11].replace(np.nan, '1', inplace= True)
new_gender = {"Gender": {"male": 1, "female": 2, "other": 3, 'unknown':3,'0':3}}
dataset_new.replace(new_gender, inplace=True)


#dataset_new.replace('unknown', np.nan, inplace= True)
dataset_new.replace('0', np.nan, inplace= True)
dataset_new.replace('No', np.nan, inplace= True)
dataset_new = pd.DataFrame(dataset_new)

X_new=dataset_new.iloc[:,[1,2,3,4,6,10]].values
Y_new = dataset_new.iloc[:,11].values
X_new = pd.DataFrame(X_new)
Y_new = pd.DataFrame(Y_new)


#for the catagorical column replace with constant
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'constant')
imputer = imputer.fit(X_new.iloc[:,[3,4]])
X_new.iloc[:,[3,4]] = imputer.transform(X_new.iloc[:,[3,4]])
 

#for the numerical values replace empty with median of the column
imputer = SimpleImputer(missing_values = np.nan,strategy = 'median')
imputer= imputer.fit(X_new.iloc[:,[0,1,2,5]].values)
X_new.iloc[:,[0,1,2,5]] = imputer.transform(X_new.iloc[:,[0,1,2,5]])

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
#X_new.iloc[:,1] = labelencoder_X.fit_transform(X_new.iloc[:,1])
X_new.iloc[:,3] = labelencoder_X.fit_transform(X_new.iloc[:,3])
#X_new.iloc[:,5] = labelencoder_X.fit_transform(X_new.iloc[:,5])
X_new.iloc[:,4] = labelencoder_X.fit_transform(X_new.iloc[:,4])
#X_new.iloc[:,7] = labelencoder_X.fit_transform(X_new.iloc[:,7])
#X_new.iloc[:,9] = labelencoder_X.fit_transform(X_new.iloc[:,9])

onehotencoder = OneHotEncoder(categorical_features =[3,4])
X_new = onehotencoder.fit_transform(X_new).toarray()


#get the prediction model

y_pred = model.predict(X_new)

y_pred = pd.DataFrame(y_pred)

y_pred = y_pred.abs()


#find which country is not in the second dataset argintina,barbados,DR congo
country1 = pd.DataFrame(X[3])
country2= pd.DataFrame(X_new[3])
country1 = country1.drop_duplicates()
country1 = country1.sort_values(by=3, ascending=True)
country2= country2.drop_duplicates()
country2 = country2.sort_values(by=3, ascending=True)

#do the same for profession

profession1 = pd.DataFrame(X[5])
profession2= pd.DataFrame(X_new[5])
profession1 = profession1.drop_duplicates()
profession1 = profession1.sort_values(by=5, ascending=True)
profession2= profession2.drop_duplicates()
profession2 = profession2.sort_values(by=5, ascending=True)
profession2.head()

Degree1 = pd.DataFrame(X[7])
Degree2= pd.DataFrame(X_new[7])
Degree1 = Degree1.drop_duplicates()
Degree1 = Degree1.sort_values(by=7, ascending=True)
Degree2= Degree2.drop_duplicates()
Degree2 = Degree2.sort_values(by=7, ascending=True)

#find unique values that are not found in test dataset
[i for i in X[5] if i not in X_new[5]]