import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split , GridSearchCV , KFold
from sklearn.preprocessing import StandardScaler 
from sklearn.feature_selection import RFE
import seaborn as sns
from sklearn.linear_model import LogisticRegression , LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB , GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix , accuracy_score , classification_report
from sklearn.svm import SVC
import matplotlib.pyplot as plt



dataset1 = pd.read_csv('G:/Software/Machine learning/Datasets/Titanic/2/full.csv')
print(dataset1.isnull().sum())


d = pd.read_csv('G:/Software/Machine learning/Datasets/Titanic/train.csv')
d1 = pd.read_csv('G:/Software/Machine learning/Datasets/Titanic/test.csv')
#dataset1 = dataset1.fillna(method = 'ffill')


dataset = dataset1.iloc[:,:12]

sns.countplot(x = 'Survived', data = dataset)

print(dataset.isnull().sum())

dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})


mean_value=dataset['Age'].mean()
dataset['Age'] = dataset['Age'].fillna(mean_value)


dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
dataset["Embarked"] = dataset["Embarked"].fillna("S")

print(round(100*(dataset.isnull().sum() / len(dataset.index)) , 2))

dataset = dataset[~np.isnan(dataset['Survived'])]

print(dataset.isnull().sum())
sns.countplot(x = 'Survived', data = dataset)


g  = sns.factorplot(x="Parch" , y="Survived" , data=dataset , kind = "bar" , size = 6 , palette = "muted")
g.despine(left=False)
g = g.set_ylabels("survival probability")


g = sns.kdeplot(dataset["Age"][(dataset["Survived"] == 0) & (dataset["Age"].notnull())], color="Red", shade = True)
g = sns.kdeplot(dataset["Age"][(dataset["Survived"] == 1) & (dataset["Age"].notnull())], ax =g, color="green", shade= True)
g.set_xlabel("Age")
g.set_ylabel("Survived Probability")
g = g.legend(["Not Survived","Survived"])


g = sns.barplot(x="Sex",y="Survived",data=dataset)
g = g.set_ylabel("Survival Probability")


dataset[["Sex","Survived"]].groupby('Sex').mean()


g = sns.factorplot(x="Pclass",y="Survived",data=dataset,kind="bar", size = 6 , palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")



g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=dataset,size=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


g = sns.factorplot(x="Pclass",y="Survived",data=dataset,kind="bar", size = 6 , palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")

g = sns.factorplot("Pclass", col="Embarked",  data=dataset,size=6, kind="count", palette="muted")
g.despine(left=True)
g = g.set_ylabels("Count")




g = sns.factorplot(y="Age",x="Sex",data=dataset,kind="box")
g = sns.factorplot(y="Age",x="Sex",hue="Pclass", data=dataset,kind="box")
g = sns.factorplot(y="Age",x="Parch", data=dataset,kind="box")
g = sns.factorplot(y="Age",x="SibSp", data=dataset,kind="box")




index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)
for i in index_NaN_age :
    age_med = dataset["Age"].median()
    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        dataset['Age'].iloc[i] = age_pred
    else :
        dataset['Age'].iloc[i] = age_med
        
        

g = sns.factorplot(x="Survived", y = "Age",data = dataset, kind="box")
g = sns.factorplot(x="Survived", y = "Age",data = dataset, kind="violin")



dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(dataset_title)
dataset["Title"].head()


g = sns.countplot(x="Title",data=dataset)
g = plt.setp(g.get_xticklabels(), rotation=45)


dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
dataset["Title"] = dataset["Title"].astype(int)


g = sns.countplot(dataset["Title"])
g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])


g = sns.factorplot(x="Title",y="Survived",data=dataset,kind="bar")
g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])
g = g.set_ylabels("survival probability")


dataset.drop(labels = ["Name"], axis = 1, inplace = True)

dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1

dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)
dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)


g = sns.factorplot(x="Single",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="SmallF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="MedF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="LargeF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")


dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])
g = sns.countplot(dataset["Cabin"],order=['A','B','C','D','E','F','G','T','X'])

g = sns.factorplot(y="Survived",x="Cabin",data=dataset,kind="bar",order=['A','B','C','D','E','F','G','T','X'])
g = g.set_ylabels("Survival Probability")

dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")

Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket.append("X")
        
dataset["Ticket"] = Ticket
dataset["Ticket"].head()


dataset = pd.get_dummies(dataset, columns = ["Ticket"] , prefix="T")

dataset["Pclass"] = dataset["Pclass"].astype("category")
dataset = pd.get_dummies(dataset, columns = ["Pclass"] , prefix="Pc")


dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)

dataset = dataset.drop(['Ticket'] , axis = 1)
dataset["Embarked"] = dataset["Embarked"].map({"C": 0, "Q":1 , "S":2})

#x = dataset.drop(['Survived' , 'Title'] , axis = 1)
x = dataset.drop(['Survived'] , axis = 1)
y = dataset['Survived']

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.20 , random_state = 90)





'''Applying ML algorithm'''

lr = LogisticRegression()
lr.fit(x_train , y_train)
y_pred_lr = lr.predict(x_test)
ac = accuracy_score(y_test , y_pred_lr)
print(ac*100)
print(confusion_matrix(y_test , y_pred_lr))
print(classification_report(y_test , y_pred_lr))


'''Knn'''
knn = KNeighborsClassifier(n_neighbors = 4)
knn.fit(x_train , y_train)

y_pred_knn = knn.predict(x_test)

ac = accuracy_score(y_test , y_pred_knn)
print(ac*100)


n_folds = 5
parameters = {
        'n_neighbors': range (2 , 50 , 2)
        }

knn = KNeighborsClassifier()

tree = GridSearchCV(estimator = knn , param_grid = parameters , cv = n_folds , n_jobs = -1)
tree.fit(x_train , y_train)

score1 = tree.cv_results_

print(pd.DataFrame(score1).head())
print(tree.best_params_)



'''SVC'''
sv = SVC()
sv.fit(x_train , y_train)

y_pred_svm = sv.predict(x_test)

ac_svr = accuracy_score(y_test , y_pred_svm)
print(ac_svr*100)


folds = KFold(n_splits = 5, shuffle = True, random_state = 4)
params = {"C": [0.01 , 0.1, 1, 10, 100, 1000]}

model = SVC()
model_cv_C = GridSearchCV(estimator = model, param_grid = params , cv = folds , verbose = 1 , return_train_score=True)
model_cv_C.fit(x_train, y_train) 


cv_results = pd.DataFrame(model_cv_C.cv_results_)
cv_results


plt.figure(figsize=(8, 6))
plt.plot(cv_results['param_C'], cv_results['mean_test_score'])
plt.plot(cv_results['param_C'], cv_results['mean_train_score'])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')


folds = KFold(n_splits = 5, shuffle = True, random_state = 4)
gamma = {'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

model = SVC()

model_cv_g = GridSearchCV(estimator = model, param_grid = gamma, cv = folds , verbose = 1 , return_train_score=True)
model_cv_g.fit(x_train, y_train) 



folds = KFold(n_splits = 5, shuffle = True, random_state = 4)
kernels = {'kernel': ['rbf' , 'poly' , 'sigmoid']}  

model = SVC()

model_cv_k = GridSearchCV(estimator = model, param_grid = kernels, cv = folds , return_train_score=True)
model_cv_k.fit(x_train, y_train) 



print(model_cv_C.best_params_)
print(model_cv_g.best_params_)
print(model_cv_k.best_params_)


sv = SVC(C = 1000 , gamma = 0.01 , kernel = 'rbf')
sv.fit(x_train , y_train)

y_pred_svm = sv.predict(x_test)

ac_svr = accuracy_score(y_test , y_pred_svm)
print(ac_svr*100)



rf = RandomForestClassifier()
rf.fit(x_train , y_train)

y_pred_rf = rf.predict(x_test)

ac_rf = accuracy_score(y_test , y_pred_rf)
print(ac_rf*100)



param_grid = {
    'max_depth': [4,6,8,10,12,15],
    'min_samples_leaf': range(1 , 100 , 2),
    'min_samples_split': range(100, 500, 100),
    'n_estimators': range(40 , 500 , 30), 
    'max_features': [1,2,3,4]
}


rf = RandomForestClassifier()
grd_search_sp = GridSearchCV(rf , param_grid , cv = folds)

grd_search_sp.fit(x_train, y_train)
print('Best parameter for min_samples_split: ',grd_search_sp.best_params_)

scores = grd_search_sp.cv_results_
pd.DataFrame(scores).head()



rf = RandomForestClassifier(n_estimators = 50 , max_depth = 8 , min_samples_leaf = 30 , max_features = 6 , min_samples_split = 60)
rf.fit(x_train , y_train)

y_pred_rf = rf.predict(x_test)

ac_rf = accuracy_score(y_test , y_pred_rf)
print(ac_rf*100)







'''Without any preprocess'''

d = pd.read_csv('train.csv')
d1 = pd.read_csv('test.csv')

d = d.drop(['Cabin'] , axis = 1)
d1 = d1.drop(['Cabin'] , axis = 1)


index_NaN_age = list(d["Age"][d["Age"].isnull()].index)
for i in index_NaN_age :
    age_med = d["Age"].median()
    age_pred = d["Age"][((d['SibSp'] == d.iloc[i]["SibSp"]) & (d['Parch'] == d.iloc[i]["Parch"]) & (d['Pclass'] == d.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        d['Age'].iloc[i] = age_pred
    else :
        d['Age'].iloc[i] = age_med;
        
        
index_NaN_age = list(d1["Age"][d1["Age"].isnull()].index)
for i in index_NaN_age :
    age_med = d1["Age"].median()
    age_pred = d1["Age"][((d1['SibSp'] == d1.iloc[i]["SibSp"]) & (d1['Parch'] == d1.iloc[i]["Parch"]) & (d1['Pclass'] == d1.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        d1['Age'].iloc[i] = age_pred
    else :
        d1['Age'].iloc[i] = age_med;
        

d1["Fare"] = d1["Fare"].fillna(d1["Fare"].median())

d["Embarked"] = d["Embarked"].fillna("S")
d["Embarked"] = d["Embarked"].map({"C": 0, "Q":1 , "S":2})

d1["Embarked"] = d1["Embarked"].fillna("S")
d1["Embarked"] = d1["Embarked"].map({"C": 0, "Q":1 , "S":2})


d["Sex"] = d["Sex"].map({"male": 0, "female":1})
d1["Sex"] = d1["Sex"].map({"male": 0, "female":1})


d = d.drop(['Name' , 'Ticket'] , axis = 1)
d1 = d1.drop(['Name' , 'Ticket'] , axis = 1)


x = d.drop(['Survived' ,'PassengerId' ,'SibSp','Parch','Embarked'] , axis = 1)
y = d['Survived']

d_test = d1.drop(['PassengerId' ,'SibSp','Parch','Embarked'] , axis = 1)

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.20 , random_state = 90)

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
d1 = sc.fit_transform(d1)

params = {'learning_rate': 0.01,
          'max_depth': 5, 
          'n_estimators': 350,
          'subsample': 0.6,}


model = XGBClassifier(params = params)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
#y_pred = model.predict(d_test)

ac_xgb = accuracy_score(y_test , y_pred)
ac_xgb


submission = pd.DataFrame({
        "PassengerId": d1["PassengerId"],
        "Survived": y_pred
    })

submission.to_csv('limontitanic02.csv', index=False)