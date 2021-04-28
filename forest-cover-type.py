"""
Created on Mon Apr 26 10:36:14 2021

Forest Cover Type Prediction
https://www.kaggle.com/c/forest-cover-type-prediction

@author: CaM
"""

############### PACKAGES ###############
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold, KFold, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

os.chdir("/Users/CaM/Desktop/Kaggle/forest-cover-type-prediction")

############### DONNEES ############### 

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

############### PREPROCESSING ############### 

df_train.info()
df_train.iloc[:,:5].describe()

df_train.isna().sum()
df_test.isna().sum()


df_train.Cover_Type.value_counts()

""" COVER TYPE CODE : 
        1 : Spruce/Fir
        2 : Lodgepole Pine
        3 : Ponderosa Pine
        4 : Cottonwood/Willow
        5 : Aspen
        6 : Douglas-fir
        7 : Krummholz
        
  => EQUILIBRE DES CLASSES : 2100 / type
"""

# Discrétisation et one hot des variables


############### VISUALISATION ###############




############### ENTRAINEMENT ############### 

data = df_train.drop("Cover_Type", axis=1)
target = df_train.Cover_Type

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=123)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Recherche des HP 

clf_lr = LogisticRegression(max_iter=1000, random_state=123)
clf_knn = KNeighborsClassifier()
clf_rf = RandomForestClassifier(n_jobs=-1, random_state=123)
clf_svm = SVC(random_state=123)

param_lr = {"solver":["liblinear", "lbfgs"], "C": np.logspace(-4,2,9)}
param_knn = {"n_neigbors": list(range(1,10)), "weights":["uniform", "distance"]}
param_rf = {"n_estimators": [10,100,1000], "min_samples_leaf": [1,3,5], "max_features":["sqrt", "log2"]}
param_svm = [{"kernel":["rbf"], "C":np.logspace(-4,4,9), "gamma":np.logspace(-4,0,9)},
             {"kernel":["linear"], "C": np.logspace(-4,4,9)}]


    # Instanciation GridSearchCV
gridcvs = {}
for param, clf, name in zip(param_rf, 
                            clf_rf, 
                            "RandomForest"):
    gvc = GridSearchCV(clf_rf, param_rf, cv=3, refit=True)
    gridcvs["name"]=gvc
    
outer_cv = StratifiedKFold(n_splits=3, shuffle=True)
    
    # Validation croisée
outer_scores = {}
for name, gs in gridcvs.items():
    nested_score= cross_val_score(gs, X_train_scaled, y_train, cv = outer_cv)
    outer_scores[name] = nested_score
    print("{} : outer accuracy {} +/- {}".format(name, (100*nested_score.mean()), (100*nested_score.std())))
    
    
    # Selection du meilleur algorithme et entraînement
final_clf = gridcvs[""]
final_clf.fit(X_train_scaled, y_train)
    


##### Utilisation VOTING CLASSIFIER #####

vclf = VotingClassifier(estimators=[("LogReg", clf_lr),
                                      ("KNN", clf_knn),
                                      ("Random Forest", clf_rf),
                                      ("SVM", clf_svm)],
                                        voting="hard")

cv3 = KFold(n_splits=3, random_state=123, shuffle=True)

clf = [clf_lr, clf_knn, clf_rf, clf_svm, vclf]
name = ["LogReg", "KNN", "Random Forest", "SVM", "VotingClassifier"]

for clf, label in zip([clf_lr, clf_knn, clf_rf, clf_svm, vclf], ["LogReg", "KNN", "Random Forest", "SVM", "VotingClassifier"]):
    score = cross_validate(clf, X_train_scaled, y_train, cv=cv3, scoring=["accuracy"])
    print("{} : \n Accuracy : {} (+/- {})"
          .format(label, 
                  np.round(score["test_accuracy"].mean(),3),
                  np.round(score["test_accuracy"].std(),3)))

##### Utilisation STACKING #####

sclf = StackingClassifier(estimators=[("LogReg", clf_lr),
                                      ("KNN", clf_knn),
                                      ("Random Forest", clf_rf),
                                      ("SVM", clf_svm)],
                                      final_estimator=clf_rf)

for clf, label in zip([clf_lr, clf_knn, clf_rf, clf_svm, sclf], ["LogReg", "KNN", "Random Forest", "SVM", "StackingClassifier"]):
    scores = cross_validate(clf, X_train_scaled, y_train, cv=cv3, scoring = ["accuracy"])
    print("{} : \n Accuracy : {} (+/- {})"
          .format(label, 
                  np.round(scores["test_accuracy"].mean(),3),
                  np.round(scores["test_accuracy"].std(),3)))


############### EVALUATION DU MODELE ############### 

final_clf = sclf.fit(X_train_scaled, y_train)

y_pred_train = final_clf.predict(X_train_scaled)
y_pred_test = final_clf.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

report = classification_report(y_test, y_pred_test, output_dict=True)
report = pd.DataFrame(report).transpose()
report

c_matrice = pd.crosstab(y_test, y_pred_test)
c_matrice = pd.DataFrame(c_matrice).transpose()









##### Utilisation XGBOOSTING #####

import xgboost as xgb

train = xgb.DMatrix(data=X_train_scaled, label=y_train)
test = xgb.DMatrix(data=X_test_scaled, label=y_test)

    # modèle 1
param_xgb = {"booster":"gbtree", "learning_rate":0.5, "objective":"binary:logistic"}
xgb_model = xgb.train(dtrain=train, params= param_xgb, num_boost_round=100, evals=[(train, "train"), (test, "eval")])

xgb_model.predict(test, ntree_limit = xgb_model.best_n_tree_limit)


    # modèle 2
params2_xgb = {"booster" : "gbtree", "learning_rate":0.01, "objective":"binary:logistic" }
xgb_model2 = xgb.train(dtrain=train, params= param_xgb, num_boost_round=100, evals=[(train, "train"), (test, "eval")])

preds = xgb_model2.predict(test, ntree_limit = xgb_model2.best_n_tree_limit)


    # Importance des features
xgb.plot_importance(xgb_model, max_num_features=15)

types=["weight", "gain", "cover", "total_gain","total_cover"]
for f in types:
    xgb.plot_importance(xgb_model, max_num_features=15, importance_type=f, title="importance : "+f);

    # Validation croisée
xgb_cv = xgb.cv(dtrain=train, params = params2_xgb, num_boost_round=100, nfold=3, early_stopping_rounds=60)

xgb_cv

    # Evaluation
xgb_preds = pd.Series(np.where(preds > 0.5,1,0))
print(pd.crosstab(xgb_preds, pd.Series(y_test)))
print(classification_report(xgb_preds, pd.Series(y_test)))






