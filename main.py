# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sk
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import r2_score
from sklearn.ensemble import VotingClassifier


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# steps:
# 1. Load CSV data, assign parameters and outputs for train and test data.
# 2. Visualize data and look at generally available trends. Normalize features.
# 3. create train test split, train logisitic regression model, possibly compile into ensemble model.
# 4. Train model, and test against test data.
train = pd.read_csv("train.csv")
test_X = pd.read_csv("test.csv")
test_id = test_X['PassengerId'].to_numpy()
y = train['Survived']
X = train.drop(['Survived','Name','Cabin'],axis=1)

X['Sex'] = LabelEncoder().fit_transform(X['Sex'])
X['Ticket'] = LabelEncoder().fit_transform(X['Ticket'])
X['Embarked'] = LabelEncoder().fit_transform(X['Embarked'])

scaler = StandardScaler()

test_X['Sex'] = LabelEncoder().fit_transform(test_X['Sex'])
test_X['Ticket'] = LabelEncoder().fit_transform(test_X['Ticket'])
test_X['Embarked'] = LabelEncoder().fit_transform(test_X['Embarked'])
test_X.drop(columns = ['PassengerId','Name','Cabin'],inplace=True)
test_X = pd.DataFrame(scaler.fit_transform(test_X.to_numpy()),columns = ['Pclass','Sex','Age','SibSp','Parch','Ticket','Fare','Embarked'] )
test_X['Sex'].fillna(test_X['Sex'].median(),inplace=True)
test_X['Pclass'].fillna(test_X['Pclass'].median(),inplace=True)
test_X['Age'].fillna(test_X['Age'].median(),inplace=True)
test_X['Parch'].fillna(test_X['Parch'].median(),inplace=True)
test_X['Ticket'].fillna(test_X['Ticket'].median(),inplace=True)
test_X['Fare'].fillna(test_X['Fare'].median(),inplace=True)
test_X['Embarked'].fillna(test_X['Embarked'].median(),inplace=True)
test_X['SibSp'].fillna(test_X['Embarked'].median(),inplace=True)





X.drop(columns = ['PassengerId'],inplace=True)
X = pd.DataFrame(scaler.fit_transform(X.to_numpy()),columns = ['Pclass','Sex','Age','SibSp','Parch','Ticket','Fare','Embarked'] )
X['Sex'].fillna(X['Sex'].median(),inplace=True)
X['Pclass'].fillna(X['Pclass'].median(),inplace=True)
X['Age'].fillna(X['Age'].median(),inplace=True)
X['Parch'].fillna(X['Parch'].median(),inplace=True)
X['Ticket'].fillna(X['Ticket'].median(),inplace=True)
X['Fare'].fillna(X['Fare'].median(),inplace=True)
X['Embarked'].fillna(X['Embarked'].median(),inplace=True)
X['SibSp'].fillna(X['Embarked'].median(),inplace=True)



X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.33, random_state=42)


logit = LogisticRegression()
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42, max_iter=1000)
NB = GaussianNB()
KN = KNeighborsClassifier()
DT = DecisionTreeClassifier()
svc = SVC()


ensemble = VotingClassifier(estimators=[('NB', NB), ('knn', KN), ('svc', svc),('lr',logit),('dt',DT),('mlp',mlp)],voting='hard', weights=[1,1,1,1,1,1])
ensemble.fit(X_train,y_train)
score0 =  ensemble.score(X_test,y_test)
y_pred_ensemble = ensemble.predict(X_test)
ensemble_r2 = r2_score(y_test,y_pred_ensemble)
print("Accuracy of Ensemble: %s"%score0)
print("r2 score of Ensemble: %s"%ensemble_r2)


logit.fit(X_train,y_train)
score1 = logit.score(X_test,y_test)
y_pred_logit = logit.predict(X_test)
logit_r2 = r2_score(y_test,y_pred_logit)
print("Accuracy of Logistic Regression: %s" %score1)
print("R2 of Logistic Regression: %s" %logit_r2)


mlp.fit(X_train,y_train)
score2 = mlp.score(X_test,y_test)
y_pred_mlp = mlp.predict(X_test)
mlp_r2 = r2_score(y_test,y_pred_mlp)
print("Accuracy of NN: %s" %score2)
print("R2 score of NN: %s" %mlp_r2)



NB.fit(X_train,y_train)
score3 = NB.score(X_test,y_test)
y_pred_NB = NB.predict(X_test)
NB_r2 = r2_score(y_test,y_pred_NB)
print("Accuracy of Naive Bayes: %s" %score3)
print("R2 score of Naive Bayes: %s" %NB_r2)



KN.fit(X_train,y_train)
score4 = KN.score(X_test,y_test)
y_pred_KN = KN.predict(X_test)
KN_r2 = r2_score(y_test,y_pred_KN)
print("Accuracy of K-Nearest: %s" %score4)
print("R2 score of K-Nearest: %s" %KN_r2)


DT.fit(X_train,y_train)
score5 = DT.score(X_test,y_test)
y_pred_DT = DT.predict(X_test)
DT_r2 = r2_score(y_test,y_pred_DT)
print("Accuracy of Decision Tree: %s" %score5)
print("R2 score of Decision Tree: %s" %DT_r2)


svc.fit(X_train,y_train)
score6 = svc.score(X_test,y_test)
y_pred_svc = svc.predict(X_test)
svc_r2 = r2_score(y_test,y_pred_svc)
print("Accuracy of Support Vector Machine: %s" %score6)
print("R2 score of Support Vector Machine: %s" %svc_r2)



Predictions = svc.predict(test_X)
Predictions = pd.DataFrame(Predictions,columns=['Survived'])
Predictions.insert(0,'PassengerId',test_id,True)

# Predictions.to_csv("C:\\Users\Riley.Powell\Desktop\TitanicPredictions.csv")
Predictions.to_csv('submission.csv',index=False)