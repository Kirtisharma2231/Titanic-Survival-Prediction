import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier

df=pd.read_csv("Data_Science/data/Titanic-Dataset.csv")
print(df.head(5))
print(df.info())

#========= Handling the missing values and dropping columns =============

print(df.isnull().sum())
print("Age : ",df.Age.median())
print("Embarked : ",df.Embarked.mode())
df['Age'].fillna(df['Age'].median(),inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)
df.drop(columns=['Cabin'],inplace=True)
print(df.info())

#==========
print(df.Survived.value_counts())
print(df.Survived.value_counts(normalize=True)*100)

#========== ploting the graph ============
sns.countplot(x='Sex',hue='Survived',data=df)
plt.show()
sns.histplot(df['Age'])
plt.show()

#========== Feature Encoding =============

df['FamilySize'] = df['SibSp'].astype(int) + df['Parch'].astype(int) + 1
df['Alone']=(df['FamilySize']==1).astype(int)
X=df.drop(columns=['PassengerId','Ticket','Name','Survived']).copy()
y=df['Survived']
X=pd.get_dummies(X,columns=['Sex','Embarked'],dtype=int,drop_first=True)
print(X.head(5))

#=========== Splitting the data into Training and Testing  ==============
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
sc=StandardScaler()
X_train_sc=sc.fit_transform(X_train)
X_test_sc=sc.transform(X_test)

#============ Comparing different Models ============
##### Model 1 -> Logistic regression #####
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
lr_model=LogisticRegression(random_state=42)
lr_model.fit(X_train_sc,y_train)
y_pred_lr=lr_model.predict(X_test_sc)
print("======== TEST DATA ACCURACY =========")
print("======== LOGISTIC REGRESSION ========")
print("Accuracy:", round(accuracy_score(y_test, y_pred_lr)*100, 2), "%")
print(classification_report(y_test, y_pred_lr))
print("Precision: ",precision_score(y_test, y_pred_lr)*100)
print("Recall: ",recall_score(y_test, y_pred_lr)*100)
print("F1: ",f1_score(y_test, y_pred_lr)*100)
print("\n")

##### Model 2 -> Decision Tree #####
from sklearn.tree import DecisionTreeClassifier
dt_model=DecisionTreeClassifier(max_depth=4,random_state=42)
dt_model.fit(X_train,y_train)
y_pred_dt=dt_model.predict(X_test)
print("======== DECISION TREE =========")
print("Decision Tree Accuracy : ",round(accuracy_score(y_test,y_pred_dt)*100,2),'%')
print("Classification Report : \n",classification_report(y_test,y_pred_dt))
print("Precision: ",precision_score(y_test,y_pred_dt)*100)
print("Recall: ",recall_score(y_test,y_pred_dt)*100)
print("F1 score: ",f1_score(y_test,y_pred_dt)*100)
print("\n")

##### Model 3 -> Bagged decision tree #####
from sklearn.ensemble import BaggingClassifier
bag=BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=5),n_estimators=200,max_samples=0.8,bootstrap=True,random_state=42)
bag.fit(X_train,y_train)
y_pred_bc=bag.predict(X_test)
print("======== BAGGING USING DECISION TREE CLASSIFIER =========")
print("Bagged Decison tree Accuracy : ",round(accuracy_score(y_test,y_pred_bc)*100,2),"%")
print("Classification Report : \n",classification_report(y_test,y_pred_bc))
print("Precision: ",precision_score(y_test,y_pred_bc)*100)
print("Recall: ",recall_score(y_test,y_pred_bc)*100)
print("F1 score: ",f1_score(y_test,y_pred_bc)*100)
print("\n")

##### Model 4 -> Random Forest Classifier #####
from sklearn.ensemble import RandomForestClassifier
rf_model=RandomForestClassifier(max_depth=None,n_estimators=200,max_features='sqrt',random_state=42)
rf_model.fit(X_train,y_train)
y_pred_rf=rf_model.predict(X_test)
print("======== RANDOM FOREST =========")
print("Random forest Accuracy : ",round(accuracy_score(y_test,y_pred_rf)*100,2),'%')
print("Classification Report : \n",classification_report(y_test,y_pred_rf))
print("Precision: ",precision_score(y_test,y_pred_rf)*100)
print("Recall: ",recall_score(y_test,y_pred_rf)*100)
print("F1 score: ",f1_score(y_test,y_pred_rf)*100)
print("\n")

###### Also getting the score for the Training data #######
print("======== TRAINING DATA ACCURACY =========")
print("Logistic Regression lr_model accuracy : ",lr_model.score(X_train_sc,y_train)*100)
print("Decision Tree dt_model accuracy : ",dt_model.score(X_train,y_train)*100)
print("Bagged Decision Tree bag accuracy : ",bag.score(X_train,y_train)*100)
print("Randon Forest rf_model accuracy : ",rf_model.score(X_train,y_train)*100)

#========= trying the model on different random data ==========
new_passenger = pd.DataFrame({
    'Pclass': [3],
    'Age': [22],
    'SibSp': [1],
    'Parch': [0],
    'Fare': [7.25],
    'FamilySize': [2],
    'Alone': [0],
    'Sex_male': [1],
    'Embarked_Q': [0],
    'Embarked_S': [1]
})

prediction = rf_model.predict(new_passenger)
print("Prediction:", prediction)