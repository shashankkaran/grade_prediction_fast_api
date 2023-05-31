import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.svm import SVC
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from collections import Counter
from sklearn import utils
from statistics import mode
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras


# DATASET READ
dataset1=pd.read_csv("cleaned.csv")
print(dataset1.info())
print(dataset1.isnull().sum())
dataset1=dataset1.fillna(0)
print(dataset1.isnull().sum())



#DATASET CLEANING & MANIPULATION


dataset1=dataset1.replace(to_replace="AB",
           value=0)

dataset1['PR'] = dataset1['PR'].astype(int)
dataset1['ENDSEM']=pd.to_numeric(dataset1['ENDSEM'])

print(dataset1['Subject'].unique())

#LABEL ENCODING AND MAPPING
encoder=LabelEncoder()
dataset1['category']=pd.cut(dataset1.ENDSEM,bins=[-1,27,55,71],labels=['Fail','First','Distinction'])
dataset1['category2']=dataset1['category'].map({'Distinction':1,'First':2,'Fail':3})
dataset1['Subject']=dataset1['Subject'].map({'DM':1, 'LD':2, 'DSA':3, 'OOP':4, 'BCN':5, 'M3':6, 'PA':7, 'DMSL':8, 'CG':9, 'SE':10, 'M1':11, 'PH':12, 'SME':13,'BE':14, 'PPS':15, 'CH':16, 'M2':17, 'SM':18, 'MECH':19, 'PHY':20})
print(dataset1)


#SPLITTING DATA INTO DEPENDENT AND INDEPENDENT FEATURES
x=dataset1.iloc[:,2:6]
y=dataset1.iloc[:,-1:]
print(x)
print(y)
dataset1.to_csv('file1.csv')
print(dataset1['category2'].value_counts())


#BALANCING THE IMBALANCED DATA 
ros=SMOTE()
xx,yy=ros.fit_resample(x,y)


#SPLITTING DATA INTO TRAING AND TESTING 
x_train,x_test,y_train,y_test=train_test_split(xx.values,yy.values.ravel(),test_size=0.2)




#DEFINING MULTIPLE ALGORITHMS
dt_clf = DecisionTreeClassifier()
knn_clf = KNeighborsClassifier()
et_clf = ExtraTreesClassifier()
gnb_classifier = GaussianNB()
lr_clf = LogisticRegression(solver='lbfgs', max_iter=10000)
rf_clf1 = RandomForestClassifier() 
lr_clf1 = LogisticRegression(solver='lbfgs', max_iter=10000)
dt_clf1 = DecisionTreeClassifier()
knn_clf1 = KNeighborsClassifier()
et_clf1 = ExtraTreesClassifier()
gnb_classifier1 = GaussianNB()
rf_clf = RandomForestClassifier() 
kf=StratifiedKFold(n_splits=11,random_state=24,shuffle=True)
classifiers =[lr_clf, dt_clf, rf_clf, et_clf, knn_clf]


#BAGGING USING MULTIPLE ALGORITHMS FOR SELECTING BEST WORKING ALGORITHM
for clf in classifiers:
    clf_scores = cross_val_score(clf, xx, yy.values.ravel(), cv =kf)
    bagging_clf = BaggingClassifier(clf, max_samples=0.4, random_state=24).fit(x_train,y_train)
    output2=bagging_clf.predict(x_test)
    output2per=metrics.accuracy_score(output2,y_test)
    print("BAGGING:",output2per)
 #   cm2 = pd.crosstab(y_test, output2)
#    print("BAGGING\n",cm2)
    bagging_clf_scores = cross_val_score(bagging_clf, xx, yy.values.ravel(), cv = kf)
    print(clf.__class__.__name__, ":::: Mean:", clf_scores.mean(), ", Std Dev:", clf_scores.std())
    print("Bagging", clf.__class__.__name__, ":::: Mean:", bagging_clf_scores.mean(), "Std Dev:", bagging_clf_scores.std(), "\n")

#USING VOTING CLASSIFIER 
ensembler = VotingClassifier(estimators=[('LogisticRegression', lr_clf1), ('DecisionTreeClassifier', dt_clf1),
                                        ('RandomForestClassifier', rf_clf1), ('ExtraTreesClassifier', et_clf1),
                                        ('KNeighborsClassifier', knn_clf1)], voting = 'hard').fit(xx.values,yy.values.ravel())

output1 = ensembler.predict(x_test)
# Save the trained model
with open('final_model.pkl', 'wb') as file:
    pickle.dump(ensembler, file)


output1per=metrics.accuracy_score(output1,y_test)
print("STACKING:",output1per)




